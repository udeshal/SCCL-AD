import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCL(nn.Module):
    def __init__(self,temperature=0.1):
        super(DCL, self).__init__()
        self.temp = temperature
    def forward(self,z,x_raw,eval=False):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        K = num_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        #loss_corr = correlation_aware_loss(z_ori.unsqueeze(1), z_trans)
        #commented for ablation 
        loss_corr = correlation_aware_loss(x_raw, z_ori)

        lamda = 0.2

        if eval:
            score = loss_tensor.sum(1) + lamda * loss_corr
            return score
        else:
            loss = loss_tensor.sum(1) + lamda * loss_corr
            return loss

class EucDCL(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temp = temperature

    def forward(self, z, eval=False):

        batch_size, num_trans, z_dim = z.shape
        sim_matrix = -torch.cdist(z,z)

        sim_matrix = torch.exp(sim_matrix / self.temp)  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        sim_matrix = sim_matrix+1e-8
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        pos_sim = sim_matrix[:, 1:, 0]

        K = num_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))
        score = (-torch.log(pos_sim) + torch.log(trans_matrix)) * scale
        if eval:
            score = score.sum(1)
            return score
        else:
            loss = score.sum(1)
            return loss

def compute_corr_matrix(x):
    """
    x: [N, D] -- N observations (here: batch samples) by D variables.
    Returns: [D, D] correlation matrix.
    """
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    x_norm = x / std
    corr = (x_norm.T @ x_norm) / x.shape[0]
    return corr


def correlation_aware_loss(x_raw, z_ori):
    """
    x_raw: [B, C]  -- the raw batch (B samples, C original variables)
    z_ori: [B, z_dim] -- the encoder's embedding of the original view, same batch

    Computes ONE correlation matrix per batch (not per individual sample),
    using the B samples in the batch as the observations.
    """
    corr_x = compute_corr_matrix(x_raw)   # [C, C]
    corr_z = compute_corr_matrix(z_ori)   # [z_dim, z_dim]

    if corr_x.shape != corr_z.shape:
        D = min(corr_x.shape[0], corr_z.shape[0])
        corr_x = corr_x[:D, :D]
        corr_z = corr_z[:D, :D]

    cos_sim = F.cosine_similarity(
        corr_x.flatten().unsqueeze(0),
        corr_z.flatten().unsqueeze(0),
        dim=1,
    )
    return 1 - cos_sim  # scalar (per-batch, not per-sample)
