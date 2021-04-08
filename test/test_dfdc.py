import sys

sys.path.insert(0, '..')
import os

if os.getcwd().endswith('test'):
    os.chdir('..')
import pathlib
import shutil
import pandas as pd
import unittest

from index_dfdc import main as index_dfdc_main
from extract_faces import main as extract_faces_main


class TestDFDC(unittest.TestCase):

    def test_1_index(self):
        df_out = 'test/data/dfdc_videos.pkl'
        if os.path.isfile(df_out):
            os.unlink(df_out)
        argv = ['--source', 'test/data/dfdc',
                '--videodataset', df_out]
        index_dfdc_main(argv)

        videos_df = pd.read_pickle(df_out)

        self.assertEqual(videos_df.shape, (6, 7))
        self.assertEqual(videos_df[videos_df.frames > 0].shape, (6, 7))

    def test_2_extract_faces(self):
        facesfolder_path = 'test/data/facecache/dfdc'
        faces_df_path = 'test/data/faces_df/dfdc_faces.pkl'
        checkpoint_path = 'test/data/tmp/dfdc'
        if os.path.isdir(facesfolder_path):
            shutil.rmtree(facesfolder_path)
        if os.path.isfile(faces_df_path):
            os.unlink(faces_df_path)
        if os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        fpv = 5
        argv = ['--source', 'test/data/dfdc',
                '--facesfolder', facesfolder_path,
                '--videodf', 'test/data/dfdc_videos.pkl',
                '--facesdf', faces_df_path,
                '--checkpoint', checkpoint_path,
                '--fpv', str(fpv),
                # '--device', 'cpu'  # TODO: remove this
                ]

        extract_faces_main(argv)

        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_0/awnfpubqmo.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_0/brtujopkby.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_1/vtfpbtmgfh.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_1/zvqinhzeah.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_10/widuwuoiur.mp4')
                                  .glob('*.jpg'))), 5)
        self.assertEqual(len(list(pathlib.Path(facesfolder_path).joinpath('dfdc_train_part_10/yhffcuhhjy.mp4')
                                  .glob('*.jpg'))), 5)

        faces_df = pd.read_pickle(faces_df_path)
        self.assertEqual(faces_df.shape, (30, 23))
