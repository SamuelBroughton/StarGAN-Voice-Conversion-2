import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader import get_loader, to_categorical


class ConditionalInstanceNormalisation(nn.Module):
    """AdaIN Block."""

    def __init__(self, dim_in, dim_c):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.gamma_t = nn.Linear(dim_c, dim_in)
        self.beta_t = nn.Linear(dim_c, dim_in)

    def forward(self, x, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        gamma = self.gamma_t(c_trg.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta_t(c_trg.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin = ConditionalInstanceNormalisation(dim_out, style_num)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c):
        x = self.conv(x)
        x = self.cin(x, c)
        x = self.glu(x)

        return x


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()
        # Down-sampling layers
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        # Bottleneck layers.
        self.residual_1 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256, dim_out=512, style_num=num_speakers)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Out.
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c)
        x = self.residual_2(x, c)
        x = self.residual_3(x, c)
        x = self.residual_4(x, c)
        x = self.residual_5(x, c)
        x = self.residual_6(x, c)
        x = self.residual_7(x, c)
        x = self.residual_8(x, c)
        x = self.residual_9(x, c)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.InstanceNorm2d(num_features=1024, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # Fully connected layer.
        self.fully_connected = nn.Linear(in_features=512, out_features=1)

        # Projection.
        self.projection = nn.Linear(2*self.num_speakers, 512)

    def forward(self, x, c, c_):
        c_onehot = torch.cat((c, c_), dim=1).to(self.device)

        x = self.conv_layer_1(x)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x_ = self.down_sample_4(x)

        h = torch.sum(x_, dim=(2, 3))
        x = self.fully_connected(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (b, 1, h, w)
        p = self.projection(c_onehot)  # (b, 512)

        in_prod = p * h

        x = x.view(x.size(0), -1)
        x = torch.mean(x, dim=-1) + torch.mean(in_prod, dim=-1)

        return x


# Just for testing shapes of architecture.
if __name__ == '__main__':
    train_dir = '../data/VCC2018-Data/mc/train'
    num_speakers = 4
    speakers_using = ['VCC2SM1', 'VCC2SM2', 'VCC2SF1', 'VCC2SF2']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    generator = Generator(num_speakers=num_speakers).to(device)
    discriminator = Discriminator(num_speakers=num_speakers).to(device)

    # Load data
    train_loader = get_loader(speakers_using, train_dir, 8, 'train', num_workers=1)
    data_iter = iter(train_loader)

    mc_real, _, spk_c_org = next(data_iter)
    mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

    spk_c = np.random.randint(0, num_speakers, size=mc_real.size(0))
    spk_c_cat = to_categorical(spk_c, num_speakers)
    spk_c_trg = torch.FloatTensor(spk_c_cat)

    mc_real = mc_real.to(device)              # Input mc.
    spk_c_org = spk_c_org.to(device)          # Original spk acc conditioning.
    spk_c_trg = spk_c_trg.to(device)          # Target spk conditioning.

    print('------------------------')
    print('Testing Discriminator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    dis_real = discriminator(mc_real, spk_c_org, spk_c_trg)
    print(f'Shape out: {dis_real.shape}')
    print('------------------------')

    print('Testing Generator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    mc_fake = generator(mc_real, spk_c_trg)
    print(f'Shape out: {mc_fake.shape}')
    print('------------------------')
