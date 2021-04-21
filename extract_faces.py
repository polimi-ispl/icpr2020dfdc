"""
Extract faces

Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.cuda
from PIL import Image
from tqdm import tqdm

import blazeface
from blazeface import BlazeFace, VideoReader, FaceExtractor
from isplutils.utils import adapt_bb


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Videos root directory', required=True)
    parser.add_argument('--videodf', type=Path, help='Path to read the videos DataFrame', required=True)
    parser.add_argument('--facesfolder', type=Path, help='Faces output root directory', required=True)
    parser.add_argument('--facesdf', type=Path, help='Path to save the output DataFrame of faces', required=True)
    parser.add_argument('--checkpoint', type=Path, help='Path to save the temporary per-video outputs', required=True)

    parser.add_argument('--fpv', type=int, default=32, help='Frames per video')
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                        help='Device to use for face extraction')
    parser.add_argument('--collateonly', help='Only perform collation of pre-existing results', action='store_true')
    parser.add_argument('--noindex', help='Do not rebuild the index', action='store_false')
    parser.add_argument('--batch', type=int, help='Batch size', default=16)
    parser.add_argument('--threads', type=int, help='Number of threads', default=8)
    parser.add_argument('--offset', type=int, help='Offset to start extraction', default=0)
    parser.add_argument('--num', type=int, help='Number of videos to process', default=0)
    parser.add_argument('--lazycheck', action='store_true', help='Lazy check of existing video indexes')
    parser.add_argument('--deepcheck', action='store_true', help='Try to open every image')

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    ## Parameters parsing
    device: torch.device = args.device
    source_dir: Path = args.source
    facedestination_dir: Path = args.facesfolder
    frames_per_video: int = args.fpv
    videodataset_path: Path = args.videodf
    facesdataset_path: Path = args.facesdf
    collateonly: bool = args.collateonly
    batch_size: int = args.batch
    threads: int = args.threads
    offset: int = args.offset
    num: int = args.num
    lazycheck: bool = args.lazycheck
    deepcheck: bool = args.deepcheck
    checkpoint_folder: Path = args.checkpoint
    index_enable: bool = args.noindex

    ## Parameters
    face_size = 512

    print('Loading video DataFrame')
    df_videos = pd.read_pickle(videodataset_path)

    if num > 0:
        df_videos_process = df_videos.iloc[offset:offset + num]
    else:
        df_videos_process = df_videos.iloc[offset:]

    if not collateonly:

        ## Blazeface loading
        print('Loading face extractor')
        facedet = BlazeFace().to(device)
        facedet.load_weights("blazeface/blazeface.pth")
        facedet.load_anchors("blazeface/anchors.npy")
        videoreader = VideoReader(verbose=False)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn, facedet)

        ## Face extraction
        with ThreadPoolExecutor(threads) as p:
            for batch_idx0 in tqdm(np.arange(start=0, stop=len(df_videos_process), step=batch_size),
                                   desc='Extracting faces'):
                tosave_list = list(p.map(partial(process_video,
                                                 source_dir=source_dir,
                                                 facedestination_dir=facedestination_dir,
                                                 checkpoint_folder=checkpoint_folder,
                                                 face_size=face_size,
                                                 face_extractor=face_extractor,
                                                 lazycheck=lazycheck,
                                                 deepcheck=deepcheck,
                                                 ),
                                         df_videos_process.iloc[batch_idx0:batch_idx0 + batch_size].iterrows()))

                for tosave in tosave_list:
                    if tosave is not None:
                        if len(tosave[2]):
                            list(p.map(save_jpg, tosave[2]))
                        tosave[1].parent.mkdir(parents=True, exist_ok=True)
                        tosave[0].to_pickle(str(tosave[1]))

    if index_enable:
        # Collect checkpoints
        df_videos['nfaces'] = np.zeros(len(df_videos), np.uint8)
        faces_dataset = []
        for idx, record in tqdm(df_videos.iterrows(), total=len(df_videos), desc='Collecting faces results'):
            # Checkpoint
            video_face_checkpoint_path = checkpoint_folder.joinpath(record['path']).with_suffix('.faces.pkl')
            if video_face_checkpoint_path.exists():
                try:
                    df_video_faces = pd.read_pickle(str(video_face_checkpoint_path))
                    # Fix same attribute issue
                    df_video_faces = df_video_faces.rename(columns={'subject': 'videosubject'}, errors='ignore')
                    nfaces = len(
                        np.unique(df_video_faces.index.map(lambda x: int(x.split('_subj')[1].split('.jpg')[0]))))
                    df_videos.loc[idx, 'nfaces'] = nfaces
                    faces_dataset.append(df_video_faces)
                except Exception as e:
                    print('Error while reading: {}'.format(video_face_checkpoint_path))
                    print(e)
                    video_face_checkpoint_path.unlink()

        if len(faces_dataset) == 0:
            raise ValueError(f'No checkpoint found from face extraction. '
                             f'Is the the source path {source_dir} correct for the videos in your dataframe?')

        # Save videos with updated faces
        print('Saving videos DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

        if offset > 0:
            if num > 0:
                if facesdataset_path.is_dir():
                    facesdataset_path = facesdataset_path.joinpath(
                        'faces_df_from_video_{}_to_video_{}.pkl'.format(offset, num + offset))
                else:
                    facesdataset_path = facesdataset_path.parent.joinpath(
                        str(facesdataset_path.parts[-1]).split('.')[0] + '_from_video_{}_to_video_{}.pkl'.format(offset,
                                                                                                                 num + offset))
            else:
                if facesdataset_path.is_dir():
                    facesdataset_path = facesdataset_path.joinpath('faces_df_from_video_{}.pkl'.format(offset))
                else:
                    facesdataset_path = facesdataset_path.parent.joinpath(
                        str(facesdataset_path.parts[-1]).split('.')[0] + '_from_video_{}.pkl'.format(offset))
        elif num > 0:
            if facesdataset_path.is_dir():
                facesdataset_path = facesdataset_path.joinpath(
                    'faces_df_from_video_{}_to_video_{}.pkl'.format(0, num))
            else:
                facesdataset_path = facesdataset_path.parent.joinpath(
                    str(facesdataset_path.parts[-1]).split('.')[0] + '_from_video_{}_to_video_{}.pkl'.format(0, num))
        else:
            if facesdataset_path.is_dir():
                facesdataset_path = facesdataset_path.joinpath('faces_df.pkl')  # just a check if the path is a dir

        # Creates directory (if doesn't exist)
        facesdataset_path.parent.mkdir(parents=True, exist_ok=True)
        print('Saving faces DataFrame to {}'.format(facesdataset_path))
        df_faces = pd.concat(faces_dataset, axis=0, )
        df_faces['video'] = df_faces['video'].astype('category')
        for key in ['kp1x', 'kp1y', 'kp2x', 'kp2y', 'kp3x',
                    'kp3y', 'kp4x', 'kp4y', 'kp5x', 'kp5y', 'kp6x', 'kp6y', 'left',
                    'top', 'right', 'bottom', ]:
            df_faces[key] = df_faces[key].astype(np.int16)
        df_faces['videosubject'] = df_faces['videosubject'].astype(np.int8)
        # Eventually remove duplicates
        df_faces = df_faces.loc[~df_faces.index.duplicated(keep='first')]
        fields_to_preserve_from_video = [i for i in
                                         ['folder', 'subject', 'scene', 'cluster', 'nfaces', 'test'] if
                                         i in df_videos]
        df_faces = pd.merge(df_faces, df_videos[fields_to_preserve_from_video], left_on='video',
                            right_index=True)
        df_faces.to_pickle(str(facesdataset_path))

    print('Completed!')


def save_jpg(args: Tuple[Image.Image, Path or str]):
    image, path = args
    image.save(path, quality=95, subsampling='4:4:4')


def process_video(item: Tuple[pd.Index, pd.Series],
                  source_dir: Path,
                  facedestination_dir: Path,
                  checkpoint_folder: Path,
                  face_size: int,
                  face_extractor: FaceExtractor,
                  lazycheck: bool = False,
                  deepcheck: bool = False,
                  ) -> (pd.DataFrame, Path, List[Tuple[Image.Image, Path]]) or None:
    # Instatiate Index and Series
    idx, record = item

    # Checkpoint
    video_faces_checkpoint_path = checkpoint_folder.joinpath(record['path']).with_suffix('.faces.pkl')

    if not lazycheck:
        if video_faces_checkpoint_path.exists():
            try:
                df_video_faces = pd.read_pickle(str(video_faces_checkpoint_path))
                for _, r in df_video_faces.iterrows():
                    face_path = facedestination_dir.joinpath(r.name)
                    assert (face_path.exists())
                    if deepcheck:
                        img = Image.open(face_path)
                        img_arr = np.asarray(img)
                        assert (img_arr.ndim == 3)
                        assert (np.prod(img_arr.shape) > 0)
            except Exception as e:
                print('Error while checking: {}'.format(video_faces_checkpoint_path))
                print(e)
                video_faces_checkpoint_path.unlink()

    if not (video_faces_checkpoint_path.exists()):

        try:

            video_face_dict_list = []

            # Load faces
            current_video_path = source_dir.joinpath(record['path'])
            if not current_video_path.exists():
                raise FileNotFoundError(f'Unable to find {current_video_path}.'
                                        f'Are you sure that {source_dir} is the correct source directory for the video '
                                        f'you indexed in the dataframe?')

            frames = face_extractor.process_video(current_video_path)

            if len(frames) == 0:
                return

            face_extractor.keep_only_best_face(frames)
            for frame_idx, frame in enumerate(frames):
                frames[frame_idx]['subjects'] = [0] * len(frames[frame_idx]['detections'])

            # Extract and save faces, bounding boxes, keypoints
            images_to_save: List[Tuple[Image.Image, Path]] = []
            for frame_idx, frame in enumerate(frames):
                if len(frames[frame_idx]['detections']):
                    fullframe = Image.fromarray(frames[frame_idx]['frame'])

                    # Preserve the only found face even if not a good one, otherwise preserve only clusters > -1
                    subjects = np.unique(frames[frame_idx]['subjects'])
                    if len(subjects) > 1:
                        subjects = np.asarray([s for s in subjects if s > -1])

                    for face_idx, _ in enumerate(frame['faces']):
                        subj_id = frames[frame_idx]['subjects'][face_idx]
                        if subj_id in subjects:  # Exclude outliers if other faces detected
                            face_path = facedestination_dir.joinpath(record['path'], 'fr{:03d}_subj{:1d}.jpg'.format(
                                frames[frame_idx]['frame_idx'], subj_id))

                            face_dict = {'facepath': str(face_path.relative_to(facedestination_dir)), 'video': idx,
                                         'label': record['label'], 'videosubject': subj_id,
                                         'original': record['original']}
                            # add attibutes for ff++
                            if 'class' in record.keys():
                                face_dict.update({'class': record['class']})
                            if 'source' in record.keys():
                                face_dict.update({'source': record['source']})
                            if 'quality' in record.keys():
                                face_dict.update({'quality': record['quality']})

                            for field_idx, key in enumerate(blazeface.BlazeFace.detection_keys):
                                face_dict[key] = frames[frame_idx]['detections'][face_idx][field_idx]

                            cropping_bb = adapt_bb(frame_height=fullframe.height,
                                                   frame_width=fullframe.width,
                                                   bb_height=face_size,
                                                   bb_width=face_size,
                                                   left=face_dict['xmin'],
                                                   top=face_dict['ymin'],
                                                   right=face_dict['xmax'],
                                                   bottom=face_dict['ymax'])
                            face = fullframe.crop(cropping_bb)

                            for key in blazeface.BlazeFace.detection_keys:
                                if (key[0] == 'k' and key[-1] == 'x') or (key[0] == 'x'):
                                    face_dict[key] -= cropping_bb[0]
                                elif (key[0] == 'k' and key[-1] == 'y') or (key[0] == 'y'):
                                    face_dict[key] -= cropping_bb[1]

                            face_dict['left'] = face_dict.pop('xmin')
                            face_dict['top'] = face_dict.pop('ymin')
                            face_dict['right'] = face_dict.pop('xmax')
                            face_dict['bottom'] = face_dict.pop('ymax')

                            face_path.parent.mkdir(parents=True, exist_ok=True)
                            images_to_save.append((face, face_path))

                            video_face_dict_list.append(face_dict)

            if len(video_face_dict_list) > 0:

                df_video_faces = pd.DataFrame(video_face_dict_list)
                df_video_faces.index = df_video_faces['facepath']
                del df_video_faces['facepath']

                # type conversions
                for key in ['kp1x', 'kp1y', 'kp2x', 'kp2y', 'kp3x', 'kp3y',
                            'kp4x', 'kp4y', 'kp5x', 'kp5y', 'kp6x', 'kp6y', 'left', 'top',
                            'right', 'bottom']:
                    df_video_faces[key] = df_video_faces[key].astype(np.int16)
                df_video_faces['conf'] = df_video_faces['conf'].astype(np.float32)
                df_video_faces['video'] = df_video_faces['video'].astype('category')

                video_faces_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            else:
                print('No faces extracted for video {}'.format(record['path']))
                df_video_faces = pd.DataFrame()

            return df_video_faces, video_faces_checkpoint_path, images_to_save

        except Exception as e:
            print('Error while processing: {}'.format(record['path']))
            print("-" * 60)
            traceback.print_exc(file=sys.stdout, limit=5)
            print("-" * 60)
            return


if __name__ == '__main__':
    main(sys.argv[1:])
