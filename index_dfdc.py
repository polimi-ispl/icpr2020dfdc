"""
Index the official Kaggle training dataset and prepares a train and validation set based on folders

Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import sys
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from isplutils.utils import extract_meta_av


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir', required=True)
    parser.add_argument('--videodataset', type=Path, default='data/dfdc_videos.pkl',
                        help='Path to save the videos DataFrame')
    parser.add_argument('--batch', type=int, help='Batch size', default=64)

    return parser.parse_args(argv)


def main(argv):
    ## Parameters parsing
    args = parse_args(argv)
    source_dir: Path = args.source
    videodataset_path: Path = args.videodataset
    batch_size: int = args.batch

    ## DataFrame
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path)
    else:
        print('Creating video DataFrame')

        # Create ouptut folder
        videodataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Index
        df_train_list = list()
        for idx, json_path in enumerate(tqdm(sorted(source_dir.rglob('metadata.json')), desc='Indexing')):
            df_tmp = pd.read_json(json_path, orient='index')
            df_tmp['path'] = df_tmp.index.map(
                lambda x: str(json_path.parent.relative_to(source_dir).joinpath(x)))
            df_tmp['folder'] = int(str(json_path.parts[-2]).split('_')[-1])
            df_train_list.append(df_tmp)
        df_videos = pd.concat(df_train_list, axis=0, verify_integrity=True)

        # Save space
        del df_videos['split']
        df_videos['label'] = df_videos['label'] == 'FAKE'
        df_videos['original'] = df_videos['original'].astype('category')
        df_videos['folder'] = df_videos['folder'].astype(np.uint8)

        # Collect metadata
        paths_arr = np.asarray(df_videos.path.map(lambda x: str(source_dir.joinpath(x))))
        height_list = []
        width_list = []
        frames_list = []
        with Pool() as pool:
            for batch_idx0 in tqdm(np.arange(start=0, stop=len(df_videos), step=batch_size), desc='Metadata'):
                batch_res = pool.map(extract_meta_av, paths_arr[batch_idx0:batch_idx0 + batch_size])
                for res in batch_res:
                    height_list.append(res[0])
                    width_list.append(res[1])
                    frames_list.append(res[2])

        df_videos['height'] = np.asarray(height_list, dtype=np.uint16)
        df_videos['width'] = np.asarray(width_list, dtype=np.uint16)
        df_videos['frames'] = np.asarray(frames_list, dtype=np.uint16)

        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))


if __name__ == '__main__':
    main(sys.argv[1:])
