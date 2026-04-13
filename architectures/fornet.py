from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from . import externals


# =========================================================
# Base Feature Extractor
# =========================================================
class FeatureExtractor(nn.Module):

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


# =========================================================
# EfficientNet Base
# =========================================================
class EfficientNetGen(FeatureExtractor):

    def __init__(self, model: str):
        super().__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)

        in_features = self.efficientnet._conv_head.out_channels

        self.classifier = nn.Linear(in_features, 1)

        # remove original FC
        if hasattr(self.efficientnet, "_fc"):
            del self.efficientnet._fc

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x  # use sigmoid outside if needed


class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super().__init__('efficientnet-b4')


# =========================================================
# EfficientNet with Attention
# =========================================================
class EfficientNetAutoAtt(EfficientNet):

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)

    def init_att(self, model: str, width: int):

        if model != 'efficientnet-b4':
            raise ValueError(f'Model not supported: {model}')

        self.att_block_idx = 9

        # dynamic channels
        in_ch = self._blocks[self.att_block_idx]._block_args.output_filters

        if width == 0:
            self.attconv = nn.Conv2d(in_ch, 1, kernel_size=1)
        else:
            layers = []
            for i in range(width):
                layers.append((f'conv{i}', nn.Conv2d(in_ch, in_ch, 3, padding=1)))
                layers.append((f'relu{i}', nn.ReLU(inplace=True)))
            layers.append(('conv_out', nn.Conv2d(in_ch, 1, kernel_size=1)))

            self.attconv = nn.Sequential(OrderedDict(layers))

        self._init_att_weights()

    def _init_att_weights(self):
        for m in self.attconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        att = None

        x = self._swish(self._bn0(self._conv_stem(x)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                break

        if att is None:
            raise RuntimeError("Attention map not computed")

        return att

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:

        x = self._swish(self._bn0(self._conv_stem(x)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                x = x * att

        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class EfficientNetGenAutoAtt(FeatureExtractor):

    def __init__(self, model: str, width: int):
        super().__init__()

        self.efficientnet = EfficientNetAutoAtt.from_pretrained(model)
        self.efficientnet.init_att(model, width)

        in_features = self.efficientnet._conv_head.out_channels
        self.classifier = nn.Linear(in_features, 1)

        if hasattr(self.efficientnet, "_fc"):
            del self.efficientnet._fc

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

    def get_attention(self, x):
        return self.efficientnet.get_attention(x)


class EfficientNetAutoAttB4(EfficientNetGenAutoAtt):
    def __init__(self):
        super().__init__('efficientnet-b4', width=0)


# =========================================================
# Xception
# =========================================================
class Xception(FeatureExtractor):

    def __init__(self):
        super().__init__()
        self.xception = externals.xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x):
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        return self.xception(x)


# =========================================================
# Siamese Tuning
# =========================================================
class SiameseTuning(FeatureExtractor):

    def __init__(self, feat_ext, num_feat, lastonly=True):
        super().__init__()

        self.feat_ext = feat_ext()

        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError("Feature extractor must implement features()")

        self.lastonly = lastonly

        # freeze backbone if needed
        if self.lastonly:
            for p in self.feat_ext.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_feat),
            nn.Linear(num_feat, 1),
        )

    def features(self, x):
        return self.feat_ext.features(x)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        return self.parameters()


class EfficientNetB4ST(SiameseTuning):
    def __init__(self):
        super().__init__(EfficientNetB4, 1792, lastonly=True)


class EfficientNetAutoAttB4ST(SiameseTuning):
    def __init__(self):
        super().__init__(EfficientNetAutoAttB4, 1792, lastonly=True)


class XceptionST(SiameseTuning):
    def __init__(self):
        super().__init__(Xception, 2048, lastonly=True)        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNet
"""


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4')


"""
EfficientNetAutoAtt
"""


class EfficientNetAutoAtt(EfficientNet):
    def init_att(self, model: str, width: int):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
            else:
                attconv_layers = []
                for i in range(width):
                    attconv_layers.append(
                        ('conv{:d}'.format(i), nn.Conv2d(kernel_size=3, padding=1, in_channels=56, out_channels=56)))
                    attconv_layers.append(
                        ('relu{:d}'.format(i), nn.ReLU(inplace=True)))
                attconv_layers.append(('conv_out', nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)))
                self.attconv = nn.Sequential(OrderedDict(attconv_layers))
        else:
            raise ValueError('Model not valid: {}'.format(model))

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        # Placeholder
        att = None

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                break

        return att

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                x = x * att

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class EfficientNetGenAutoAtt(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetGenAutoAtt, self).__init__()

        self.efficientnet = EfficientNetAutoAtt.from_pretrained(model)
        self.efficientnet.init_att(model, width)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet.get_attention(x)


class EfficientNetAutoAttB4(EfficientNetGenAutoAtt):
    def __init__(self):
        super(EfficientNetAutoAttB4, self).__init__(model='efficientnet-b4', width=0)


"""
Xception
"""


class Xception(FeatureExtractor):
    def __init__(self):
        super(Xception, self).__init__()
        self.xception = externals.xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception.forward(x)


"""
Siamese tuning
"""


class SiameseTuning(FeatureExtractor):
    def __init__(self, feat_ext: FeatureExtractor, num_feat: int, lastonly: bool = True):
        super(SiameseTuning, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError('The provided feature extractor needs to provide a features() method')
        self.lastonly = lastonly
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feat),
            nn.Linear(in_features=num_feat, out_features=1),
        )

    def features(self, x):
        x = self.feat_ext.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lastonly:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        else:
            return self.parameters()


class EfficientNetB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetB4ST, self).__init__(feat_ext=EfficientNetB4, num_feat=1792, lastonly=True)


class EfficientNetAutoAttB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetAutoAttB4ST, self).__init__(feat_ext=EfficientNetAutoAttB4, num_feat=1792, lastonly=True)


class XceptionST(SiameseTuning):
    def __init__(self):
        super(XceptionST, self).__init__(feat_ext=Xception, num_feat=2048, lastonly=True)
