import torch
import torch.nn as nn
import torchvision


def Egde(size=3, channel=1, scale=1e-3):
    if size == 3:
        param = torch.ones((channel, 1, 3, 3), dtype=torch.float32) * (-1)
        for i in range(channel):
            param[i][0][1][1] = 8
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.ones((channel, 1, 5, 5), dtype=torch.float32) * (-1)
        for i in range(channel):
            param[i][0][1][2] = 2
            param[i][0][2][1] = 2
            param[i][0][2][2] = 4
            param[i][0][2][3] = 2
            param[i][0][3][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Sobel(size=3, channel=1, scale=1e-3, direction='x'):
    if size == 3:
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = param[i][0][2][0] = 1
            param[i][0][0][2] = param[i][0][2][2] = -1
            param[i][0][1][0] = 2
            param[i][0][1][2] = -2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.zeros((channel, 1, 5, 5), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = param[i][0][4][0] = 1
            param[i][0][0][1] = param[i][0][4][1] = 2
            param[i][0][0][3] = param[i][0][4][3] = -2
            param[i][0][0][4] = param[i][0][4][4] = -1

            param[i][0][1][0] = param[i][0][3][0] = 4
            param[i][0][1][1] = param[i][0][3][1] = 8
            param[i][0][1][3] = param[i][0][3][3] = -8
            param[i][0][1][4] = param[i][0][3][4] = -4

            param[i][0][2][0] = 6
            param[i][0][2][1] = 12
            param[i][0][2][3] = -12
            param[i][0][2][4] = -6
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    if direction == 'x':
        return param
    else:
        return param.transpose(3, 2)


def Sobel_xy(size=3, channel=1, scale=1e-3, direction='xy'):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
    if size == 3 and direction == 'xy':
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 2
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -2
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'yx':
        for i in range(channel):
            param[i][0][0][0] = -2
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Roberts(size=3, channel=1, scale=1e-3, direction='x'):
    if size == 3 and direction == 'x':
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'y':
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == 'x':
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == 'y':
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Prewitt(size=3, channel=1, scale=1e-3, direction='x'):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
    if size == 3 and direction == 'y':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][1][0] = -1
            param[i][0][2][0] = -1
            param[i][0][0][2] = 1
            param[i][0][1][2] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'x':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][0][2] = -1
            param[i][0][2][0] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'xy':
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -1
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'yx':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Laplacian(channel=1, scale=1e-3, type=1):
    param = torch.ones((channel, 1, 3, 3), dtype=torch.float32)
    if type == 1:
        for i in range(channel):
            param[i][0][0][0] = 0
            param[i][0][0][2] = 0
            param[i][0][1][1] = -4
            param[i][0][2][0] = 0
            param[i][0][2][2] = 0
        param = nn.Parameter(data=param * scale, requires_grad=False)
    else:
        for i in range(channel):
            param[i][0][1][1] = -4
        param = nn.Parameter(data=param * scale, requires_grad=False)
    return param


def HighPass(x, kernel_size=15, sigma=5):
    filter2 = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    gauss = filter2(x)
    return x - gauss

