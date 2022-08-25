import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_laplacian(tensor, device="cpu"):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).to(device)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1) / 9
    return laplacian






def ncc_global(sources, targets, device='cpu', **params):
    size = sources.size(2) * sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    ncc = (1 / size) * torch.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std),
                                 dim=(1, 2, 3))

    return ncc


class NCCLoss(nn.Module):

    def __init__(self, device='cpu'):
        super(NCCLoss, self).__init__()
        self.device = device
        self.NCC = ncc_global

    def forward(self, sources, targets):
        ncc = self.NCC(sources, targets, self.device)
        # if ncc != ncc:   # 如果ncc中有nan，比如source_std=0,强行赋值为1
        #     return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
        if sources.shape[1] == 3:
            return -ncc / 3
        else:
            return -ncc


def curvature_regularization(displacement_fields, device="cpu"):
    if displacement_fields.shape[1] == 2:
        displacement_fields = displacement_fields
    elif displacement_fields.shape[-1] == 2:
        displacement_fields = displacement_fields.permute(0,3,1,2)

    u_x = displacement_fields[:, 0, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))
    u_y = displacement_fields[:, 1, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3))

    x_laplacian = tensor_laplacian(u_x, device)[:, :, 1:-1, 1:-1]
    y_laplacian = tensor_laplacian(u_y, device)[:, :, 1:-1, 1:-1]
    x_term = x_laplacian ** 2
    y_term = y_laplacian ** 2
    # print(x_term, y_term)
    curvature = torch.mean(1 / 2 * (x_term + y_term))
    return curvature
