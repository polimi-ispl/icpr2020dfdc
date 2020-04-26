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
import os
import shutil
import warnings

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import tripletnet
from train_binclass import save_model, tb_attention
from isplutils.data import FrameFaceIterableDataset
from isplutils.data_siamese import FrameFaceTripletIterableDataset
from isplutils import split, utils


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='Net model class', required=True)
    parser.add_argument('--traindb', type=str, help='Training datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--valdb', type=str, help='Validation datasets', nargs='+', choices=split.available_datasets,
                        required=True)
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
    parser.add_argument('--face', type=str, help='Face crop or scale', required=True,
                        choices=['scale', 'tight'])
    parser.add_argument('--size', type=int, help='Train patch size', required=True)

    parser.add_argument('--batch', type=int, help='Batch size to fit in GPU memory', default=12)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--valint', type=int, help='Validation interval (iterations)', default=500)
    parser.add_argument('--patience', type=int, help='Patience before dropping the LR [validation intervals]',
                        default=10)
    parser.add_argument('--maxiter', type=int, help='Maximum number of iterations', default=20000)
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')

    parser.add_argument('--traintriplets', type=int, help='Limit the number of train triplets per epoch', default=-1)
    parser.add_argument('--valtriplets', type=int, help='Limit the number of validation triplets per epoch',
                        default=2000)

    parser.add_argument('--logint', type=int, help='Training log interval (iterations)', default=100)
    parser.add_argument('--workers', type=int, help='Num workers for data loaders', default=6)
    parser.add_argument('--device', type=int, help='GPU device id', default=0)
    parser.add_argument('--seed', type=int, help='Random seed', default=0)

    parser.add_argument('--debug', action='store_true', help='Activate debug')
    parser.add_argument('--suffix', type=str, help='Suffix to default tag')

    parser.add_argument('--attention', action='store_true',
                        help='Enable Tensorboard log of attention masks')
    parser.add_argument('--embedding', action='store_true', help='Activate embedding visualization in TensorBoard')
    parser.add_argument('--embeddingint', type=int, help='Embedding visualization interval in TensorBoard',
                        default=5000)

    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default='runs/triplet/')
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default='weights/triplet/')

    args = parser.parse_args()

    # Parse arguments
    net_class = getattr(tripletnet, args.net)
    train_datasets = args.traindb
    val_datasets = args.valdb
    dfdc_df_path = args.dfdc_faces_df_path
    ffpp_df_path = args.ffpp_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir
    ffpp_faces_dir = args.ffpp_faces_dir
    face_policy = args.face
    face_size = args.size

    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience
    max_num_iterations = args.maxiter
    initial_model = args.init
    train_from_scratch = args.scratch

    max_train_triplets = args.traintriplets
    max_val_triplets = args.valtriplets

    log_interval = args.logint
    num_workers = args.workers
    device = torch.device('cuda:{:d}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    seed = args.seed

    debug = args.debug
    suffix = args.suffix

    enable_attention = args.attention
    enable_embedding = args.embedding
    embedding_interval = args.embeddingint

    weights_folder = args.models_dir
    logs_folder = args.log_dir

    # Random initialization
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load net
    net: nn.Module = net_class().to(device)

    # Loss and optimizers
    criterion = nn.TripletMarginLoss()

    min_lr = initial_lr * 1e-5
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=min_lr,
    )

    tag = utils.make_train_tag(net_class=net_class,
                               traindb=train_datasets,
                               face_policy=face_policy,
                               patch_size=face_size,
                               seed=seed,
                               suffix=suffix,
                               debug=debug,
                               )

    # Model checkpoint paths
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')

    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    # Load model
    val_loss = min_val_loss = 20
    epoch = iteration = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        net_state = state['net']
    elif not train_from_scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if net_state is not None:
        adapt_binclass_model(net_state)
        incomp_keys = net.load_state_dict(net_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = initial_lr
        optimizer.load_state_dict(opt_state)

    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(logdir, ignore_errors=True)

    # TensorboardX instance
    tb = SummaryWriter(logdir=logdir)
    if iteration == 0:
        dummy = torch.randn((1, 3, face_size, face_size), device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tb.add_graph(net, [dummy, dummy, dummy], verbose=False)

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=net.get_normalizer(), train=True)

    # Datasets and data loaders
    print('Loading data')
    # Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
    for dataset in train_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for training!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for training!')
    for dataset in val_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for validation!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for validation!')
    splits = split.make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, dfdc_dir=dfdc_faces_dir,
                               ffpp_dir=ffpp_faces_dir, dbs={'train': train_datasets, 'val': val_datasets})
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]

    train_dataset = FrameFaceTripletIterableDataset(roots=train_roots,
                                                    dfs=train_dfs,
                                                    scale=face_policy,
                                                    num_triplets=max_train_triplets,
                                                    transformer=transformer,
                                                    size=face_size,
                                                    )

    val_dataset = FrameFaceTripletIterableDataset(roots=val_roots,
                                                  dfs=val_dfs,
                                                  scale=face_policy,
                                                  num_triplets=max_val_triplets,
                                                  transformer=transformer,
                                                  size=face_size,
                                                  )

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, )

    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, )

    print('Training triplets: {}'.format(len(train_dataset)))
    print('Validation triplets: {}'.format(len(val_dataset)))

    if len(train_dataset) == 0:
        print('No training triplets. Halt.')
        return

    if len(val_dataset) == 0:
        print('No validation triplets. Halt.')
        return

    # Embedding visualization
    if enable_embedding:
        train_dataset_embedding = FrameFaceIterableDataset(roots=train_roots,
                                                           dfs=train_dfs,
                                                           scale=face_policy,
                                                           num_samples=64,
                                                           transformer=transformer,
                                                           size=face_size,
                                                           )
        train_loader_embedding = DataLoader(train_dataset_embedding, num_workers=num_workers, batch_size=batch_size, )
        val_dataset_embedding = FrameFaceIterableDataset(roots=val_roots,
                                                         dfs=val_dfs,
                                                         scale=face_policy,
                                                         num_samples=64,
                                                         transformer=transformer,
                                                         size=face_size,
                                                         )
        val_loader_embedding = DataLoader(val_dataset_embedding, num_workers=num_workers, batch_size=batch_size, )

    else:
        train_loader_embedding = None
        val_loader_embedding = None

    stop = False
    while not stop:

        # Training
        optimizer.zero_grad()

        train_loss = train_num = 0
        for train_batch in tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False,
                                total=len(train_loader) // train_loader.batch_size):
            net.train()
            train_batch_num = len(train_batch[0])
            train_num += train_batch_num

            train_batch_loss = batch_forward(net, device, criterion, train_batch)

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # Optimization
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                # Checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, last_path)
                train_loss = train_num = 0

            # Validation
            if iteration > 0 and (iteration % validation_interval == 0):

                # Validation
                val_loss = validation_routine(net, device, val_loader, criterion, tb, iteration, tag='val')
                tb.flush()

                # LR Scheduler
                lr_scheduler.step(val_loss)

                # Model checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                           periodic_path.format(iteration))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    shutil.copy(periodic_path.format(iteration), bestval_path)

                # Attention
                if enable_attention and hasattr(net, 'feat_ext') and hasattr(net.feat_ext, 'get_attention'):
                    net.eval()
                    # For each dataframe show the attention for a real,fake couple of frames

                    for df, root, sample_idx, tag in [
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == False].index[0],
                         'train/att/real'),
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == True].index[0],
                         'train/att/fake'),
                    ]:
                        record = df.loc[sample_idx]
                        tb_attention(tb, tag, iteration, net.feat_ext, device, face_size, face_policy,
                                     transformer, root, record)

                if optimizer.param_groups[0]['lr'] <= min_lr:
                    print('Reached minimum learning rate. Stopping.')
                    stop = True
                    break

            # Embedding visualization
            if enable_embedding:
                if iteration > 0 and (iteration % embedding_interval == 0):
                    embedding_routine(net=net,
                                      device=device,
                                      loader=train_loader_embedding,
                                      iteration=iteration,
                                      tb=tb,
                                      tag=tag + '/train')
                    embedding_routine(net=net,
                                      device=device,
                                      loader=val_loader_embedding,
                                      iteration=iteration,
                                      tb=tb,
                                      tag=tag + '/val')

            iteration += 1

            if iteration > max_num_iterations:
                print('Maximum number of iterations reached')
                stop = True
                break

            # End of iteration

        epoch += 1

    # Needed to flush out last events
    tb.close()

    print('Completed')


def adapt_binclass_model(net_state):
    # Check that the model contains at least one key starting with feat_ext, otherwise adapt
    found = False
    for key in net_state:
        if key.startswith('feat_ext.'):
            found = True
            break
    if not found:
        # Adapt all keys
        print('Adapting keys')
        keys = [k for k in net_state]
        for key in keys:
            net_state['feat_ext.{}'.format(key)] = net_state[key]
            del net_state[key]


def batch_forward(net: nn.Module, device, criterion, data: tuple) -> torch.Tensor:
    if torch.cuda.is_available():
        data = [i.cuda(device) for i in data]
    out = net(*data)
    loss = criterion(*out)
    return loss


def validation_routine(net, device, val_loader, criterion, tb, iteration, tag):
    net.eval()

    val_num = 0
    val_loss = 0.
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader) // val_loader.batch_size):
        val_batch_num = len(val_data[0])
        with torch.no_grad():
            val_batch_loss = batch_forward(net, device, criterion, val_data, )
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    return val_loss


def embedding_routine(net: nn.Module, device: torch.device, loader: DataLoader, tb: SummaryWriter, iteration: int,
                      tag: str):
    net.eval()

    labels = []
    embeddings = []
    for batch_data in loader:
        batch_faces, batch_labels = batch_data
        if torch.cuda.is_available():
            batch_faces = batch_faces.to(device)
        with torch.no_grad():
            batch_emb = net.features(batch_faces)
        labels.append(batch_labels.numpy().flatten())
        embeddings.append(torch.flatten(batch_emb.cpu(), start_dim=1).numpy())

    labels = list(np.concatenate(labels))
    embeddings = np.concatenate(embeddings)

    # Logging
    tb.add_embedding(mat=embeddings, metadata=labels, tag=tag, global_step=iteration)


if __name__ == '__main__':
    main()
