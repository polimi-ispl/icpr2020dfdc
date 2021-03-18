"""
Index Celeb-DF v2
Image and Sound Processing Lab - Politecnico di Milano
Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from isplutils.utils import extract_meta_av, extract_meta_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir',
                        required=True)
    parser.add_argument('--videodataset', type=Path, default='data/celebdf_videos.pkl',
                        help='Path to save the videos DataFrame')

    args = parser.parse_args()

    ## Parameters parsing
    source_dir: Path = args.source
    videodataset_path: Path = args.videodataset

    # Create ouput folder (if doesn't exist)
    videodataset_path.parent.mkdir(parents=True, exist_ok=True)

    ## DataFrame
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path)
    else:
        print('Creating video DataFrame')

        split_file = Path(source_dir).joinpath('List_of_testing_videos.txt')
        if not split_file.exists():
            raise FileNotFoundError('Unable to find "List_of_testing_videos.txt" in {}'.format(source_dir))
        test_videos_df = pd.read_csv(split_file, delimiter=' ', header=0, index_col=1)

        ff_videos = Path(source_dir).rglob('*.mp4')
        df_videos = pd.DataFrame(
            {'path': [f.relative_to(source_dir) for f in ff_videos]})

        df_videos['height'] = df_videos['width'] = df_videos['frames'] = np.zeros(len(df_videos), dtype=np.uint16)
        with Pool() as p:
            meta = p.map(extract_meta_av, df_videos['path'].map(lambda x: str(source_dir.joinpath(x))))
        meta = np.stack(meta)
        df_videos.loc[:, ['height', 'width', 'frames']] = meta

        # Fix for videos that av cannot decode properly
        for idx, record in df_videos[df_videos['frames'] == 0].iterrows():
            meta = extract_meta_cv(str(source_dir.joinpath(record['path'])))
            df_videos.loc[idx, ['height', 'width', 'frames']] = meta

        df_videos['class'] = df_videos['path'].map(lambda x: x.parts[0]).astype('category')
        df_videos['label'] = df_videos['class'].map(
            lambda x: True if x == 'Celeb-synthesis' else False)  # True is FAKE, False is REAL
        df_videos['name'] = df_videos['path'].map(lambda x: x.with_suffix('').name)

        df_videos['original'] = -1 * np.ones(len(df_videos), dtype=np.int16)
        df_videos.loc[(df_videos['label'] == True), 'original'] = \
            df_videos[(df_videos['label'] == True)]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == '_'.join([x.split('_')[0], x.split('_')[2]]))[0]]
            )

        df_videos['test'] = df_videos['path'].map(str).isin(test_videos_df.index)

        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))


if __name__ == '__main__':
    main()
