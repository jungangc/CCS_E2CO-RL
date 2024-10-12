import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def fc_bn_relu(hidden_dim):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    )

def conv_bn_relu(in_filter, out_filter, nb_row, nb_col, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_filter, out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1)),
        nn.BatchNorm2d(out_filter),
        nn.ReLU()
    )

class ResidualConv(nn.Module):
    def __init__(self, in_filter, out_filter, nb_row, nb_col, stride=(1, 1)):
        super(ResidualConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_filter)
        self.conv2 = nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_filter)

    def forward(self, x):
        identity = x.clone()

        a = self.conv1(x)
        a = self.bn1(a)
        a = F.relu(a)

        a = self.conv2(a)
        a = self.bn2(a)

        y = identity + a

        return y

def dconv_bn_nolinear(in_filter, out_filter, nb_row, nb_col, stride=(2, 2), activation="relu", padding=0):
    return nn.Sequential(
#         nn.ConvTranspose2d(in_filter, out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=(1, 1)),
        nn.ConvTranspose2d(in_filter, out_filter, kernel_size=(nb_row, nb_col), stride=stride, padding=padding),
        nn.BatchNorm2d(out_filter),
        nn.ReLU()
    )

class ReflectionPadding2D(nn.Module):
    def __init__(self, padding=(1, 1)):
        super(ReflectionPadding2D, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), 'reflect')

class UnPooling2D(nn.Module):
    def __init__(self, size=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.size, mode='nearest')