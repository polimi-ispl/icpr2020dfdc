from pprint import pprint
from typing import Iterable, List

import albumentations as A
import cv2
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn as nn
from torchvision import transforms


def adapt_bb(frame_height: int, frame_width: int, bb_height: int, bb_width: int, left: int, top: int, right: int,
             bottom: int) -> (
        int, int, int, int):
    x_ctr = (left + right) // 2
    y_ctr = (bottom + top) // 2
    new_top = max(y_ctr - bb_height // 2, 0)
    new_bottom = min(new_top + bb_height, frame_height)
    new_left = max(x_ctr - bb_width // 2, 0)
    new_right = min(new_left + bb_width, frame_width)
    return new_left, new_top, new_right, new_bottom


def extract_bb(frame: Image.Image, bb: Iterable, scale: str, size: int) -> Image.Image:
    """
    Extract a face from a frame according to the given bounding box and scale policy
    :param frame: Entire frame
    :param bb: Bounding box (left,top,right,bottom) in the reference system of the frame
    :param scale: "scale" to crop a square with size equal to the maximum between height and width of the face, then scale to size
                  "crop" to crop a fixed square around face center,
                  "tight" to crop face exactly at the bounding box with no scaling
    :param size: size of the face
    :return:
    """
    left, top, right, bottom = bb
    if scale == "scale":
        bb_width = int(right) - int(left)
        bb_height = int(bottom) - int(top)
        bb_to_desired_ratio = min(size / bb_height, size / bb_width) if (bb_width > 0 and bb_height > 0) else 1.
        bb_width = int(size / bb_to_desired_ratio)
        bb_height = int(size / bb_to_desired_ratio)
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bb_height, bb_width, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
    elif scale == "crop":
        # Find the center of the bounding box and cut an area around it of height x width
        left, top, right, bottom = adapt_bb(frame.height, frame.width, size, size, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    elif scale == "tight":
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bottom - top, right - left, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    else:
        raise ValueError('Unknown scale value: {}'.format(scale))

    return face


def showimage(img_tensor: torch.Tensor):
    topil = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0, ], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])
    plt.figure()
    plt.imshow(topil(img_tensor))
    plt.show()


def make_train_tag(net_class: nn.Module,
                   face_policy: str,
                   patch_size: int,
                   traindb: List[str],
                   seed: int,
                   suffix: str,
                   debug: bool,
                   ):
    # Training parameters and tag
    tag_params = dict(net=net_class.__name__,
                      traindb='-'.join(traindb),
                      face=face_policy,
                      size=patch_size,
                      seed=seed
                      )
    print('Parameters')
    pprint(tag_params)
    tag = 'debug_' if debug else ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    if suffix is not None:
        tag += '_' + suffix
    print('Tag: {:s}'.format(tag))
    return tag


def get_transformer(face_policy: str,
                    patch_size: int,
                    net_normalizer: transforms.Normalize):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0),
        ]
        downsample_train_transformations = [
            A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
        ]
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0),
        ]
        downsample_train_transformations = [
            A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
        ]
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    # Only in training
    aug_transformations = [
        A.Compose([
            A.HorizontalFlip(),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
            ]),
            A.OneOf([
                A.ISONoise(),
                A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
            ]),
            A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
            A.ImageCompression(quality_lower=50, quality_upper=99),
        ], )
    ]

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf
