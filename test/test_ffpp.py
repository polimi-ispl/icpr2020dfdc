import sys

sys.path.insert(0, '..')
import os

if os.getcwd().endswith('test'):
    os.chdir('..')
import pathlib
import shutil
import pandas as pd
import unittest

from index_ffpp import main as index_ffpp_main
from extract_faces import main as extract_faces_main


class TestFFPP(unittest.TestCase):

    def test_1_index(self):
        df_out = 'test/data/ffpp_videos.pkl'
        if os.path.isfile(df_out):
            os.unlink(df_out)
        argv = ['--source', 'test/data/ffpp',
                '--videodataset', df_out]
        index_ffpp_main(argv)

        videos_df = pd.read_pickle(df_out)

        self.assertEqual(videos_df.shape, (10, 10))
        self.assertEqual(videos_df[videos_df.frames > 0].shape, (10, 10))

    def test_2_extract_faces(self):
        facesfolder_path = 'test/data/facecache/ffpp'
        faces_df_path = 'test/data/faces_df/ffpp_faces.pkl'
        checkpoint_path = 'test/data/tmp/ffpp'
        if os.path.isdir(facesfolder_path):
            shutil.rmtree(facesfolder_path)
        if os.path.isfile(faces_df_path):
            os.unlink(faces_df_path)
        if os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        fpv = 5
        argv = ['--source', 'test/data/ffpp',
                '--facesfolder', facesfolder_path,
                '--videodf', 'test/data/ffpp_videos.pkl',
                '--facesdf', faces_df_path,
                '--checkpoint', checkpoint_path,
                '--fpv', str(fpv),
                # '--device', 'cuda:5'  # TODO: remove this
                ]

        extract_faces_main(argv)

        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('manipulated_sequences/DeepFakeDetection/c23/videos/'
                                            '24_23__outside_talking_still_laughing__YR5OVD4S.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('manipulated_sequences/Deepfakes/c23/videos/'
                                            '519_515.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('manipulated_sequences/Face2Face/c23/videos/'
                                            '750_743.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('manipulated_sequences/FaceSwap/c23/videos/'
                                            '634_660.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('manipulated_sequences/NeuralTextures/c23/videos/'
                                            '004_982.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('original_sequences/actors/c23/videos/'
                                            '24__outside_talking_still_laughing.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('original_sequences/youtube/c23/videos/'
                                            '004.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('original_sequences/youtube/c23/videos/'
                                            '519.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('original_sequences/youtube/c23/videos/'
                                            '634.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path)
                                  .joinpath('original_sequences/youtube/c23/videos/'
                                            '750.mp4')
                                  .glob('*.jpg'))), 5)

        faces_df = pd.read_pickle(faces_df_path)
        self.assertEqual(faces_df.shape, (50, 25))
