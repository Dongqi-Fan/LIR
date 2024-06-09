from einops import rearrange
import torch.nn.functional as F
from archs.filters import *
import torch
import torch.nn as nn
from utils.registry import ARCH_REGISTRY
from thop import profile
from einops import rearrange

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False, activation='prelu',
                 group=1, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias, groups=group)

        self.norm = norm
        if self.norm == 'BN':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'IN':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'gelu':
            self.act = torch.nn.GELU()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Filters(nn.Module):
    def __init__(self, dim, training=True):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        self.training = training
        self.dim = dim
        self.Sobel_x = Sobel(channel=dim, direction='x').to(device)
        self.Sobel_y = Sobel(channel=dim, direction='y').to(device)
        self.Laplation = Laplacian(channel=dim).to(device)
        self.Edge = Egde(channel=dim).to(device)
        self.Roberts_x = Roberts(channel=dim, direction='x').to(device)
        self.Roberts_y = Roberts(channel=dim, direction='y').to(device)
        self.Sobel_xy = Sobel_xy(channel=dim, direction='xy').to(device)
        self.Sobel_yx = Sobel_xy(channel=dim, direction='yx').to(device)
        self.alpha = nn.Parameter(torch.ones_like(torch.FloatTensor(9)).to(device).requires_grad_())
        self.beta = nn.Parameter(torch.zeros_like(torch.FloatTensor(1)).to(device).requires_grad_())
        self.weight = None

        if not self.training:
            self.weight = self.Sobel_x * self.alpha[0] + self.Sobel_y * self.alpha[1] + self.Laplation * self.alpha[2] + \
                          self.Edge * self.alpha[3] + self.Roberts_x * self.alpha[4] + self.Roberts_y * self.alpha[5] + \
                          self.Sobel_xy * self.alpha[6] + self.Sobel_yx * self.alpha[7]
            self.__delattr__('Sobel_x')
            self.__delattr__('Sobel_y')
            self.__delattr__('Laplation')
            self.__delattr__('Edge')
            self.__delattr__('Roberts_x')
            self.__delattr__('Roberts_y')
            self.__delattr__('Sobel_xy')
            self.__delattr__('Sobel_yx')

    def forward(self, x):
        if self.weight is None:
            Sobel_x = F.conv2d(input=x, weight=self.Sobel_x, stride=1, groups=self.dim, padding=1) * self.alpha[0]
            Sobel_y = F.conv2d(input=x, weight=self.Sobel_y, stride=1, groups=self.dim, padding=1) * self.alpha[1]
            Laplation = F.conv2d(input=x, weight=self.Laplation, stride=1, groups=self.dim, padding=1) * self.alpha[2]
            Egde = F.conv2d(input=x, weight=self.Edge, stride=1, groups=self.dim, padding=1) * self.alpha[3]
            Sobel_xy = F.conv2d(input=x, weight=self.Sobel_xy, stride=1, groups=self.dim, padding=1) * self.alpha[4]
            Sobel_yx = F.conv2d(input=x, weight=self.Sobel_yx, stride=1, groups=self.dim, padding=1) * self.alpha[5]
            Roberts_x = F.conv2d(input=x, weight=self.Roberts_x, stride=1, groups=self.dim, padding=1) * self.alpha[6]
            Roberts_y = F.conv2d(input=x, weight=self.Roberts_y, stride=1, groups=self.dim, padding=1) * self.alpha[7]
            high_pass = HighPass(x) * self.alpha[8]
            return Sobel_x + Sobel_y + Laplation + Egde + x * self.beta[
                0] + Sobel_xy + Sobel_yx + Roberts_x + Roberts_y + high_pass
        else:
            out = F.conv2d(input=x, weight=self.weight, stride=1, groups=self.dim, padding=1)
            return HighPass(x) * self.alpha[8] + out + x * self.beta[0]


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def __init__(self, dim):
        super(SimpleGate, self).__init__()
        self.norm = LayerNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim * 2, 1, padding=0)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = res + x * self.gamma
        return x


class PatchAttention(nn.Module):
    def __init__(self, patch_size, dim, mlp_dim, bias=False):
        super().__init__()
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]
        self.down = nn.Conv2d(dim, mlp_dim, 1, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(self.patch_h * self.patch_w * mlp_dim, self.patch_h * self.patch_w),
            nn.LayerNorm(self.patch_h * self.patch_w),
            nn.GELU(),
            nn.Linear(self.patch_h * self.patch_w, 1),
        )
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        b, c, h, w = x.shape
        assert h % self.patch_h == 0, 'Image hight must be divisible by the patch_h.'
        assert w % self.patch_w == 0, 'Image width must be divisible by the patch_w.'
        h_scale = h // self.patch_h
        w_scale = w // self.patch_w
        y = self.down(x)
        y = rearrange(y, 'b c (h hs) (w ws) -> b (hs ws) (c h w)', hs=h_scale, ws=w_scale)
        x = rearrange(x, 'b c (h hs) (w ws) -> b (hs ws) (c h w)', hs=h_scale, ws=w_scale)
        y = self.mlp(y)
        x = x * y
        x = rearrange(x, 'b (hs ws) (c ph pw) -> b c (ph hs) (pw ws)', hs=h_scale, ws=w_scale, ph=self.patch_h,
                      pw=self.patch_w)
        x = self.beta * res + x
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.conv_2 = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
        self.layernorm = LayerNorm2d(dim * 2)
        self.act = nn.GELU()
        self.conv_3 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.alpha = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.layernorm(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = res + x * self.alpha
        return x


class ResCABlock(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.conv_2 = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
        self.layernorm = LayerNorm2d(dim * 2)
        self.act = nn.GELU()
        self.conv_3 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.ca = ChannelAttention(dim)
        self.alpha = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.layernorm(x)
        x = self.act(x)
        x = self.conv_3(x)
        x = self.ca(x)
        x = res + x * self.alpha
        return x


class AttentionBlock(nn.Module):
    def __init__(self, patch_size, dim, mlp_dim, bias=False):
        super().__init__()
        self.resblock = ResBlock(dim, bias)
        self.pa = PatchAttention(patch_size, dim, mlp_dim, bias)
        self.sg = SimpleGate(dim)
        self.rescablock = ResCABlock(dim, bias)

    def forward(self, x):
        x = self.resblock(x)
        x = self.pa(x)
        x = self.sg(x)
        x = self.rescablock(x)
        return x


class EAABlock(nn.Module):
    def __init__(self, dim, patch_size, dim_scale, bias=False, training=True):
        super().__init__()
        self.training = training
        self.stage1 = nn.Sequential(
            AttentionBlock((patch_size, patch_size), dim, dim // dim_scale),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=bias),
        )

        self.stage2 = nn.Sequential(
            Filters(dim, self.training),
            AttentionBlock((patch_size // 2, patch_size // 2), dim, dim // dim_scale),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=bias),
        )

        self.stage3 = nn.Sequential(
            Filters(dim, self.training),
            AttentionBlock((patch_size // 2, patch_size // 2), dim, dim // dim_scale),
        )
  
        self.stage4 = nn.Sequential(
            Filters(dim, self.training),
            AttentionBlock((patch_size // 2, patch_size // 2), dim, dim // dim_scale),
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias),
        )

        self.stage5 = nn.Sequential(
            Filters(dim * 2, self.training),
            AttentionBlock((patch_size, patch_size), dim * 2, dim * 2 // dim_scale),
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1, bias=bias),
        )

        self.transpose = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=bias)

    def forward(self, x):
        res = x
        lq = self.stage1(x)
        bottle = self.stage3(self.stage2(lq))
        hq = self.stage4(bottle)
        x = self.stage5(torch.cat([hq, lq], dim=1))
        degradation = self.transpose(lq - hq)
        return x + res - degradation

@ARCH_REGISTRY.register()
class LIR(nn.Module):
    def __init__(self, img_channel=3, dim=48, left_blk_num=[3, 3], bottom_blk_num=3, right_blk_num=[4, 3], training=True):
        super().__init__()
        self.training = training
        self.Conv_head = ConvBlock(img_channel, dim)

        self.EAA1 = nn.Sequential(*[EAABlock(dim, 8, 16, training=self.training) for i in range(left_blk_num[0])])
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.EAA2 = nn.Sequential(*[EAABlock(dim, 8, 8, training=self.training) for i in range(left_blk_num[1])])
        self.downsample2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.EAA3 = nn.Sequential(*[EAABlock(dim, 4, 8, training=self.training) for i in range(bottom_blk_num)])
        self.upsample2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.down = ConvBlock(dim * 2, dim)

        self.EAA4 = nn.Sequential(*[EAABlock(dim, 8, 8, training=self.training) for i in range(right_blk_num[1])])
        self.upsample1 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.EAA5 = nn.Sequential(*[EAABlock(dim, 8, 16, training=self.training) for i in range(right_blk_num[0])])
        self.transpose1 = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.transpose2 = nn.ConvTranspose2d(dim, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.Conv_tail = ConvBlock(dim, img_channel)

    def forward(self, x):
        img = x
        x = self.Conv_head(x)
        res = x

        x = self.EAA1(x)
        lq = self.downsample1(x)

        x = self.EAA2(lq)
        x = self.downsample2(x)

        x = self.EAA3(x)
        x = self.upsample2(x)
        x = self.down(torch.cat([x, lq], dim=1))

        hq = self.EAA4(x)
        x = self.upsample1(hq)

        x = self.EAA5(x) + res - self.transpose1(lq - hq)
        x = self.Conv_tail(x) + img - self.transpose2(lq - hq)
        return x 


if __name__ == '__main__':
    from thop import profile
    model = LIR(training=False)
    input = torch.randn(1, 3, 192, 192)
    flops, _ = profile(model, inputs=(input,))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 7.31 M
# Total gflops: 62.40

