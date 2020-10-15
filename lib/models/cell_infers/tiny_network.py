#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch.nn as nn
from ..cell_operations import ResNetBasicblock
from .cells import InferCell


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):

  def __init__(self, C, N, genotype, num_classes):
    super(TinyNetwork, self).__init__()
    self._C               = C
    self._layerN          = N

    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    self.layer_reductions = layer_reductions

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2, True)
      else:
        cell = InferCell(genotype, C_prev, C_curr, 1)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer= len(self.cells)

    self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):

    list_features = []

    feature = self.stem(inputs)
    list_features.append(feature)

    for i, cell in enumerate(self.cells):
      feature = cell(feature)

      #if i == len(self.layer_reductions)-1 or self.layer_reductions[i+1]:
      list_features.append(feature)
      #  continue
      #if i % 2 == 0:
      #  list_features.append(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    list_features.append(out)

    logits = self.classifier(out)

    return list_features, logits
