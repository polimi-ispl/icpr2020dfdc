import os
from typing import Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

from blazeface import BlazeFace


class FaceExtractor:
    """Wrapper for face extraction workflow."""

    def __init__(self, video_read_fn = None, facedet: BlazeFace = None):
        """Creates a new FaceExtractor.

        Arguments:
            video_read_fn: a function that takes in a path to a video file
                and returns a tuple consisting of a NumPy array with shape
                (num_frames, H, W, 3) and a list of frame indices, or None
                in case of an error
            facedet: the face detector object
        """
        self.video_read_fn = video_read_fn
        self.facedet = facedet

    def process_image(self, path: str = None, img: Image.Image or np.ndarray = None) -> dict:
        """
        Process a single image
        :param path: Path to the image
        :param img: image
        :return:
        """

        if img is not None and path is not None:
            raise ValueError('Only one argument between path and img can be specified')
        if img is None and path is None:
            raise ValueError('At least one argument between path and img must be specified')

        target_size = self.facedet.input_size

        if img is None:
            img = np.asarray(Image.open(str(path)))
        else:
            img = np.asarray(img)

        # Split the frames into several tiles. Resize the tiles to 128x128.
        tiles, resize_info = self._tile_frames(np.expand_dims(img, 0), target_size)
        # tiles has shape (num_tiles, target_size, target_size, 3)
        # resize_info is a list of four elements [resize_factor_y, resize_factor_x, 0, 0]

        # Run the face detector. The result is a list of PyTorch tensors,
        # one for each tile in the batch.
        detections = self.facedet.predict_on_batch(tiles, apply_nms=False)

        # Convert the detections from 128x128 back to the original frame size.
        detections = self._resize_detections(detections, target_size, resize_info)

        # Because we have several tiles for each frame, combine the predictions
        # from these tiles. The result is a list of PyTorch tensors, but now one
        # for each frame (rather than each tile).
        num_frames = 1
        frame_size = (img.shape[1], img.shape[0])
        detections = self._untile_detections(num_frames, frame_size, detections)

        # The same face may have been detected in multiple tiles, so filter out
        # overlapping detections. This is done separately for each frame.
        detections = self.facedet.nms(detections)

        # Crop the faces out of the original frame.
        frameref_detections = self._add_margin_to_detections(detections[0], frame_size, 0.2)
        faces = self._crop_faces(img, frameref_detections)
        kpts = self._crop_kpts(img, detections[0], 0.3)

        # Add additional information about the frame and detections.
        scores = list(detections[0][:, 16].cpu().numpy())
        frame_dict = {"frame_w": frame_size[0],
                      "frame_h": frame_size[1],
                      "faces": faces,
                      "kpts": kpts,
                      "detections": frameref_detections.cpu().numpy(),
                      "scores": scores,
                      }

        # Sort faces by descending confidence
        frame_dict = self._soft_faces_by_descending_score(frame_dict)

        return frame_dict

    def _soft_faces_by_descending_score(self, frame_dict: dict) -> dict:
        if len(frame_dict['scores']) > 1:
            sort_idxs = np.argsort(frame_dict['scores'])[::-1]
            new_faces = [frame_dict['faces'][i] for i in sort_idxs]
            new_kpts = [frame_dict['kpts'][i] for i in sort_idxs]
            new_detections = frame_dict['detections'][sort_idxs]
            new_scores = [frame_dict['scores'][i] for i in sort_idxs]
            frame_dict['faces'] = new_faces
            frame_dict['kpts'] = new_kpts
            frame_dict['detections'] = new_detections
            frame_dict['scores'] = new_scores
        return frame_dict

    def process_videos(self, input_dir, filenames, video_idxs) -> List[dict]:
        """For the specified selection of videos, grabs one or more frames
        from each video, runs the face detector, and tries to find the faces
        in each frame.

        The frames are split into tiles, and the tiles from the different videos
        are concatenated into a single batch. This means the face detector gets
        a batch of size len(video_idxs) * num_frames * num_tiles (usually 3).

        Arguments:
            input_dir: base folder where the video files are stored
            filenames: list of all video files in the input_dir
            video_idxs: one or more indices from the filenames list; these
                are the videos we'll actually process

        Returns a list of dictionaries, one for each frame read from each video.

        This dictionary contains:
            - video_idx: the video this frame was taken from
            - frame_idx: the index of the frame in the video
            - frame_w, frame_h: original dimensions of the frame
            - faces: a list containing zero or more NumPy arrays with a face crop
            - scores: a list array with the confidence score for each face crop

        If reading a video failed for some reason, it will not appear in the
        output array. Note that there's no guarantee a given video will actually
        have num_frames results (as soon as a reading problem is encountered for
        a video, we continue with the next video).
        """
        target_size = self.facedet.input_size

        videos_read = []
        frames_read = []
        frames = []
        tiles = []
        resize_info = []

        for video_idx in video_idxs:
            # Read the full-size frames from this video.
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)

            # Error? Then skip this video.
            if result is None: continue

            videos_read.append(video_idx)

            # Keep track of the original frames (need them later).
            my_frames, my_idxs = result
            frames.append(my_frames)
            frames_read.append(my_idxs)

            # Split the frames into several tiles. Resize the tiles to 128x128.
            my_tiles, my_resize_info = self._tile_frames(my_frames, target_size)
            tiles.append(my_tiles)
            resize_info.append(my_resize_info)

        if len(tiles) == 0:
            return []
        # Put all the tiles for all the frames from all the videos into
        # a single batch.
        batch = np.concatenate(tiles)

        # Run the face detector. The result is a list of PyTorch tensors,
        # one for each image in the batch.
        all_detections = self.facedet.predict_on_batch(batch, apply_nms=False)

        result = []
        offs = 0
        for v in range(len(tiles)):
            # Not all videos may have the same number of tiles, so find which
            # detections go with which video.
            num_tiles = tiles[v].shape[0]
            detections = all_detections[offs:offs + num_tiles]
            offs += num_tiles

            # Convert the detections from 128x128 back to the original frame size.
            detections = self._resize_detections(detections, target_size, resize_info[v])

            # Because we have several tiles for each frame, combine the predictions
            # from these tiles. The result is a list of PyTorch tensors, but now one
            # for each frame (rather than each tile).
            num_frames = frames[v].shape[0]
            frame_size = (frames[v].shape[2], frames[v].shape[1])
            detections = self._untile_detections(num_frames, frame_size, detections)

            # The same face may have been detected in multiple tiles, so filter out
            # overlapping detections. This is done separately for each frame.
            detections = self.facedet.nms(detections)

            for i in range(len(detections)):
                # Crop the faces out of the original frame.
                frameref_detections = self._add_margin_to_detections(detections[i], frame_size, 0.2)
                faces = self._crop_faces(frames[v][i], frameref_detections)
                kpts = self._crop_kpts(frames[v][i], detections[i], 0.3)

                # Add additional information about the frame and detections.
                scores = list(detections[i][:, 16].cpu().numpy())
                frame_dict = {"video_idx": videos_read[v],
                              "frame_idx": frames_read[v][i],
                              "frame_w": frame_size[0],
                              "frame_h": frame_size[1],
                              "frame": frames[v][i],
                              "faces": faces,
                              "kpts": kpts,
                              "detections": frameref_detections.cpu().numpy(),
                              "scores": scores,
                              }
                # Sort faces by descending confidence
                frame_dict = self._soft_faces_by_descending_score(frame_dict)

                result.append(frame_dict)

        return result

    def process_video(self, video_path):
        """Convenience method for doing face extraction on a single video."""
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self.process_videos(input_dir, filenames, [0])

    def _tile_frames(self, frames: np.ndarray, target_size: Tuple[int, int]) -> (np.ndarray, List[float]):
        """Splits each frame into several smaller, partially overlapping tiles
        and resizes each tile to target_size.

        After a bunch of experimentation, I found that for a 1920x1080 video,
        BlazeFace works better on three 1080x1080 windows. These overlap by 420
        pixels. (Two windows also work but it's best to have a clean center crop
        in there as well.)

        I also tried 6 windows of size 720x720 (horizontally: 720|360, 360|720;
        vertically: 720|1200, 480|720|480, 1200|720) but that gives many false
        positives when a window has no face in it.

        For a video in portrait orientation (1080x1920), we only take a single
        crop of the top-most 1080 pixels. If we split up the video vertically,
        then we might get false positives again.

        (NOTE: Not all videos are necessarily 1080p but the code can handle this.)

        Arguments:
            frames: NumPy array of shape (num_frames, height, width, 3)
            target_size: (width, height)

        Returns:
            - a new (num_frames * N, target_size[1], target_size[0], 3) array
              where N is the number of tiles used.
            - a list [scale_w, scale_h, offset_x, offset_y] that describes how
              to map the resized and cropped tiles back to the original image
              coordinates. This is needed for scaling up the face detections
              from the smaller image to the original image, so we can take the
              face crops in the original coordinate space.
        """
        num_frames, H, W, _ = frames.shape

        num_h, num_v, split_size, x_step, y_step = self.get_tiles_params(H, W)

        splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y:y + split_size, x:x + split_size, :]
                    splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
        return splits, resize_info

    def get_tiles_params(self, H, W):
        split_size = min(H, W, 720)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = (H - split_size) // y_step + 1 if y_step > 0 else 1
        num_h = (W - split_size) // x_step + 1 if x_step > 0 else 1
        return num_h, num_v, split_size, x_step, y_step

    def _resize_detections(self, detections, target_size, resize_info):
        """Converts a list of face detections back to the original
        coordinate system.

        Arguments:
            detections: a list containing PyTorch tensors of shape (num_faces, 17)
            target_size: (width, height)
            resize_info: [scale_w, scale_h, offset_x, offset_y]
        """
        projected = []
        target_w, target_h = target_size
        scale_w, scale_h, offset_x, offset_y = resize_info

        for i in range(len(detections)):
            detection = detections[i].clone()

            # ymin, xmin, ymax, xmax
            for k in range(2):
                detection[:, k * 2] = (detection[:, k * 2] * target_h - offset_y) * scale_h
                detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_w - offset_x) * scale_w

            # keypoints are x,y
            for k in range(2, 8):
                detection[:, k * 2] = (detection[:, k * 2] * target_w - offset_x) * scale_w
                detection[:, k * 2 + 1] = (detection[:, k * 2 + 1] * target_h - offset_y) * scale_h

            projected.append(detection)

        return projected

    def _untile_detections(self, num_frames: int, frame_size: Tuple[int, int], detections: List[torch.Tensor]) -> List[
        torch.Tensor]:
        """With N tiles per frame, there also are N times as many detections.
        This function groups together the detections for a given frame; it is
        the complement to tile_frames().
        """
        combined_detections = []

        W, H = frame_size

        num_h, num_v, split_size, x_step, y_step = self.get_tiles_params(H, W)

        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    # Adjust the coordinates based on the split positions.
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k * 2] += y
                            detection[:, k * 2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k * 2] += x
                            detection[:, k * 2 + 1] += y

                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step

            combined_detections.append(torch.cat(detections_for_frame))

        return combined_detections

    def _add_margin_to_detections(self, detections: torch.Tensor, frame_size: Tuple[int, int],
                                  margin: float = 0.2) -> torch.Tensor:
        """Expands the face bounding box.

        NOTE: The face detections often do not include the forehead, which
        is why we use twice the margin for ymin.

        Arguments:
            detections: a PyTorch tensor of shape (num_detections, 17)
            frame_size: maximum (width, height)
            margin: a percentage of the bounding box's height

        Returns a PyTorch tensor of shape (num_detections, 17).
        """
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset * 2, min=0)  # ymin
        detections[:, 1] = torch.clamp(detections[:, 1] - offset, min=0)  # xmin
        detections[:, 2] = torch.clamp(detections[:, 2] + offset, max=frame_size[1])  # ymax
        detections[:, 3] = torch.clamp(detections[:, 3] + offset, max=frame_size[0])  # xmax
        return detections

    def _crop_faces(self, frame: np.ndarray, detections: torch.Tensor) -> List[np.ndarray]:
        """Copies the face region(s) from the given frame into a set
        of new NumPy arrays.

        Arguments:
            frame: a NumPy array of shape (H, W, 3)
            detections: a PyTorch tensor of shape (num_detections, 17)

        Returns a list of NumPy arrays, one for each face crop. If there
        are no faces detected for this frame, returns an empty list.
        """
        faces = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(int)
            face = frame[ymin:ymax, xmin:xmax, :]
            faces.append(face)
        return faces

    def _crop_kpts(self, frame: np.ndarray, detections: torch.Tensor, face_fraction: float):
        """Copies the parts region(s) from the given frame into a set
        of new NumPy arrays.

        Arguments:
            frame: a NumPy array of shape (H, W, 3)
            detections: a PyTorch tensor of shape (num_detections, 17)
            face_fraction: float between 0 and 1 indicating how big are the parts to be extracted w.r.t the whole face

        Returns a list of NumPy arrays, one for each face crop. If there
        are no faces detected for this frame, returns an empty list.
        """
        faces = []
        for i in range(len(detections)):
            kpts = []
            size = int(face_fraction * min(detections[i, 2] - detections[i, 0], detections[i, 3] - detections[i, 1]))
            kpts_coords = detections[i, 4:16].cpu().numpy().astype(int)
            for kpidx in range(6):
                kpx, kpy = kpts_coords[kpidx * 2:kpidx * 2 + 2]
                kpt = frame[kpy - size // 2:kpy - size // 2 + size, kpx - size // 2:kpx - size // 2 + size, ]
                kpts.append(kpt)
            faces.append(kpts)
        return faces

    def remove_large_crops(self, crops, pct=0.1):
        """Removes faces from the results if they take up more than X%
        of the video. Such a face is likely a false positive.

        This is an optional postprocessing step. Modifies the original
        data structure.

        Arguments:
            crops: a list of dictionaries with face crop data
            pct: maximum portion of the frame a crop may take up
        """
        for i in range(len(crops)):
            frame_data = crops[i]
            video_area = frame_data["frame_w"] * frame_data["frame_h"]
            faces = frame_data["faces"]
            scores = frame_data["scores"]
            new_faces = []
            new_scores = []
            for j in range(len(faces)):
                face = faces[j]
                face_H, face_W, _ = face.shape
                face_area = face_H * face_W
                if face_area / video_area < 0.1:
                    new_faces.append(face)
                    new_scores.append(scores[j])
            frame_data["faces"] = new_faces
            frame_data["scores"] = new_scores

    def keep_only_best_face(self, crops):
        """For each frame, only keeps the face with the highest confidence.

        This gets rid of false positives, but obviously is problematic for
        videos with two people!

        This is an optional postprocessing step. Modifies the original
        data structure.
        """
        for i in range(len(crops)):
            frame_data = crops[i]
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]

    # TODO: def filter_likely_false_positives(self, crops):
    #   if only some frames have more than 1 face, it's likely a false positive
    #   if most frames have more than 1 face, it's probably two people
    #   so find the % of frames with > 1 face; if > 0.X, keep the two best faces

    # TODO: def filter_by_score(self, crops, min_score) to remove any
    # crops with a confidence score lower than min_score

    # TODO: def sort_by_histogram(self, crops) for videos with 2 people.
