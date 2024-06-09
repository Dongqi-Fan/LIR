import torch
from torch import nn as nn
from torch.nn import functional as F
import random
from torchvision import models

from utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@weighted_loss
def content_loss(pred, target):
    return torch.square(pred - target).mean()


@weighted_loss
def style_loss(pred, target):
    return torch.square(pred - target).mean() * (-1)

@LOSS_REGISTRY.register()
class ContentLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(ContentLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * content_loss(pred, target, weight)


@LOSS_REGISTRY.register()
class StyleLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(StyleLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * style_loss(pred, target, weight)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) \
                                  * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class FeatureLoss(nn.Module):
    def __init__(self, loss=nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs) == len(targets)
        length = len(outputs)

        stu_ch = outputs[0].size()[1]
        tea_ch = targets[0].size()[1]
        index = random.randint(0, tea_ch - stu_ch)
        for i in range(length):
            targets[i] = targets[i][:, index:index+stu_ch, :, :]

        for i in range(length):
            outputs[i] = spatial_similarity(outputs[i])
        for i in range(length):
            targets[i] = spatial_similarity(targets[i])

        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss


def spatial_similarity(x):
    fm = x.cpu()
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1).cuda()
    return s


class Vgg19(torch.nn.Module):
    def __init__(self, weight_file, requires_grad=False):
        super(Vgg19, self).__init__()
        #vgg_pretrained_features = models.vgg19(pretrained=True).features
        device_id = torch.cuda.current_device()
        model = models.vgg19(pretrained=False)
        model.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage.cuda(device_id)))
        vgg_pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    

@LOSS_REGISTRY.register()
class ContrastLoss(nn.Module):
    def __init__(self, weights, d_func, weight_file, t_detach=False, is_one=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19(weight_file).cuda()
        self.l1 = nn.L1Loss()
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach
        self.weights = []
        weights = weights.split(',')
        for i in range(len(weights)):
            self.weights.append(float(weights[i]))

    def forward(self, teacher, student, neg, blur_neg=None):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        blur_neg_vgg = None
        if blur_neg is not None:
            blur_neg_vgg = self.vgg(blur_neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward
        elif self.d_func == 'cos':
            self.forward_func = self.cos_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, blur_neg_vgg)

    def L1_forward(self, teacher, student, neg, blur_neg=None):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))

            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive
        return loss

    def cos_forward(self, teacher, student, neg, blur_neg=None):
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))

            if self.t_detach:
                d_ts = torch.cosine_similarity(teacher[i].detach(), student[i], dim=0).mean()
            else:
                d_ts = torch.cosine_similarity(teacher[i], student[i], dim=0).mean()
            d_sn = self.calc_cos_stu_neg(student[i], neg_i.detach())

            contrastive = -torch.log(torch.exp(d_ts)/(torch.exp(d_sn)+1e-7))
            loss += self.weights[i] * contrastive
        return loss

    def calc_cos_stu_neg(self, stu, neg):
        n = stu.shape[0]
        m = neg.shape[0]

        stu = stu.view(n, -1)
        neg = neg.view(m, n, -1)
        # normalize
        stu = F.normalize(stu, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)
        # multiply
        d_sn = torch.mean((stu * neg).sum(0))
        return d_sn