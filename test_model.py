"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import argparse
import gc
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import fornet
from architectures.fornet import FeatureExtractor
from isplutils import utils, split
from isplutils.data import FrameFaceDatasetTest


def main():
    # Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--testsets', type=str, help='Testing datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--testsplits', type=str, help='Test split', nargs='+', default=['val', 'test'],
                        choices=['train', 'val', 'test'])
    parser.add_argument('--dfdc_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--dfdc_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parser.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')

    # Specify trained model path
    parser.add_argument('--model_path', type=Path, help='Full path of the trained model', required=True)

    # Common params
    parser.add_argument('--batch', type=int, help='Batch size to fit in GPU memory', default=128)

    parser.add_argument('--workers', type=int, help='Num workers for data loaders', default=6)
    parser.add_argument('--device', type=int, help='GPU id', default=0)

    parser.add_argument('--debug', action='store_true', help='Debug flag', )
    parser.add_argument('--num_video', type=int, help='Number of real-fake videos to test')
    parser.add_argument('--results_dir', type=Path, help='Output folder',
                        default='results/')

    parser.add_argument('--override', action='store_true', help='Override existing results', )

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    num_workers: int = args.workers
    batch_size: int = args.batch
    max_num_videos_per_label: int = args.num_video  # number of real-fake videos to test
    model_path: Path = args.model_path
    results_dir: Path = args.results_dir
    debug: bool = args.debug
    override: bool = args.override
    test_sets = args.testsets
    test_splits = args.testsplits
    dfdc_df_path = args.dfdc_faces_df_path
    ffpp_df_path = args.ffpp_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir
    ffpp_faces_dir = args.ffpp_faces_dir

    # get arguments from the model path
    face_policy = str(model_path).split('face-')[1].split('_')[0]
    patch_size = int(str(model_path).split('size-')[1].split('_')[0])
    net_name = str(model_path).split('net-')[1].split('_')[0]
    model_name = '_'.join(model_path.with_suffix('').parts[-2:])

    # Load net
    net_class = getattr(fornet, net_name)

    # load model
    print('Loading model...')
    state_tmp = torch.load(model_path, map_location='cpu')
    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    net: FeatureExtractor = net_class().eval().to(device)

    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!')

    # val loss per-frame
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Define data transformers
    test_transformer = utils.get_transformer(face_policy, patch_size, net.get_normalizer(), train=False)

    # datasets and dataloaders (from train_binclass.py)
    print('Loading data...')
    # Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
    for dataset in test_sets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for testing!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for testing!')
    splits = split.make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, dfdc_dir=dfdc_faces_dir,
                               ffpp_dir=ffpp_faces_dir, dbs={'train': test_sets, 'val': test_sets, 'test': test_sets})
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]
    test_dfs = [splits['test'][db][0] for db in splits['test']]
    test_roots = [splits['test'][db][1] for db in splits['test']]

    # Output paths
    out_folder = results_dir.joinpath(model_name)
    out_folder.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Samples selection
    if max_num_videos_per_label and max_num_videos_per_label > 0:
        dfs_out_train = [select_videos(df, max_num_videos_per_label) for df in train_dfs]
        dfs_out_val = [select_videos(df, max_num_videos_per_label) for df in val_dfs]
        dfs_out_test = [select_videos(df, max_num_videos_per_label) for df in test_dfs]
    else:
        dfs_out_train = train_dfs
        dfs_out_val = val_dfs
        dfs_out_test = test_dfs

    # Extractions list
    extr_list = []
    # Append train and validation set first
    if 'train' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append(
                (dfs_out_train[idx], out_folder.joinpath(dataset + '_train.pkl'), train_roots[idx], dataset + ' TRAIN')
            )
    if 'val' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append(
                (dfs_out_val[idx], out_folder.joinpath(dataset + '_val.pkl'), val_roots[idx], dataset + ' VAL')
            )
    if 'test' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append(
                (dfs_out_test[idx], out_folder.joinpath(dataset + '_test.pkl'), test_roots[idx], dataset + ' TEST')
            )

    for df, df_path, df_root, tag in extr_list:
        if override or not df_path.exists():
            print('\n##### PREDICT VIDEOS FROM {} #####'.format(tag))
            print('Real frames: {}'.format(sum(df['label'] == False)))
            print('Fake frames: {}'.format(sum(df['label'] == True)))
            print('Real videos: {}'.format(df[df['label'] == False]['video'].nunique()))
            print('Fake videos: {}'.format(df[df['label'] == True]['video'].nunique()))
            dataset_out = process_dataset(root=df_root, df=df, net=net, criterion=criterion,
                                          patch_size=patch_size,
                                          face_policy=face_policy, transformer=test_transformer,
                                          batch_size=batch_size,
                                          num_workers=num_workers, device=device, )
            df['score'] = dataset_out['score'].astype(np.float32)
            df['loss'] = dataset_out['loss'].astype(np.float32)
            print('Saving results to: {}'.format(df_path))
            df.to_pickle(str(df_path))

            if debug:
                plt.figure()
                plt.title(tag)
                plt.hist(df[df.label == True].score, bins=100, alpha=0.6, label='FAKE frames')
                plt.hist(df[df.label == False].score, bins=100, alpha=0.6, label='REAL frames')
                plt.legend()

            del (dataset_out)
            del (df)
            gc.collect()

    if debug:
        plt.show()

    print('Completed!')


def process_dataset(df: pd.DataFrame,
                    root: str,
                    net: FeatureExtractor,
                    criterion,
                    patch_size: int,
                    face_policy: str,
                    transformer: A.BasicTransform,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    ) -> dict:
    if isinstance(device, (int, str)):
        device = torch.device(device)

    dataset = FrameFaceDatasetTest(
        root=root,
        df=df,
        size=patch_size,
        scale=face_policy,
        transformer=transformer,
    )

    # Preallocate
    score = np.zeros(len(df))
    loss = np.zeros(len(df))

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    with torch.no_grad():
        idx0 = 0
        for batch_data in tqdm(loader):
            batch_images = batch_data[0].to(device)
            batch_labels = batch_data[1].to(device)
            batch_samples = len(batch_images)
            batch_out = net(batch_images)
            batch_loss = criterion(batch_out, batch_labels)
            score[idx0:idx0 + batch_samples] = batch_out.cpu().numpy()[:, 0]
            loss[idx0:idx0 + batch_samples] = batch_loss.cpu().numpy()[:, 0]
            idx0 += batch_samples

    out_dict = {'score': score, 'loss': loss}
    return out_dict


def select_videos(df: pd.DataFrame, max_videos_per_label: int) -> pd.DataFrame:
    """
    Select up to a maximum number of videos
    :param df: DataFrame of frames. Required columns: 'video','label'
    :param max_videos_per_label: maximum number of real and fake videos
    :return: DataFrame of selected frames
    """
    # Save random state
    st0 = np.random.get_state()
    # Set seed for this selection only
    np.random.seed(42)

    df_fake = df[df.label == True]
    fake_videos = df_fake['video'].unique()
    selected_fake_videos = np.random.choice(fake_videos, min(max_videos_per_label, len(fake_videos)), replace=False)
    df_selected_fake_frames = df_fake[df_fake['video'].isin(selected_fake_videos)]

    df_real = df[df.label == False]
    real_videos = df_real['video'].unique()
    selected_real_videos = np.random.choice(real_videos, min(max_videos_per_label, len(real_videos)), replace=False)
    df_selected_real_frames = df_real[df_real['video'].isin(selected_real_videos)]
    # Restore random state
    np.random.set_state(st0)

    return pd.concat((df_selected_fake_frames, df_selected_real_frames), axis=0, verify_integrity=True).copy()


if __name__ == '__main__':
    main()
