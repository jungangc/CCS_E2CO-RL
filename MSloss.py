import numpy as np
import h5py
import torch
import torch.nn as nn

class CustomizedLoss(nn.Module):
    def __init__(self, lambda_flux_loss, lambda_bhp_loss, lambda_trans_loss, lambda_yobs_loss):
        super(CustomizedLoss, self).__init__()
        self.flux_loss_lambda = lambda_flux_loss
        self.bhp_loss_lambda = lambda_bhp_loss
        self.trans_loss_weight = lambda_trans_loss
        self.yobs_loss_weight = lambda_yobs_loss

    def forward(self, pred):
        # Parse y_pred
        # X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec, perm = pred
        X_next_pred, X_next, Z_next_pred, Z_next, Yobs_pred, Yobs, z0, x0, x0_rec = pred

        # loss_rec_t = get_reconstruction_loss(xt, xt_rec)
        loss_rec_t = get_reconstruction_loss(x0, x0_rec)
        # loss_flux_t = get_flux_loss(perm, x0, x0_rec) * self.flux_loss_lambda
        loss_flux_t = 0
        loss_prod_bhp_t=0
        loss_rec_t1 = 0
        loss_flux_t1 =0
        loss_prod_bhp_t1 = 0
        for x_next, x_next_pred in zip(X_next, X_next_pred):
            loss_rec_t1 += get_reconstruction_loss(x_next, x_next_pred)
            # loss_flux_t1 += get_flux_loss(perm, x_next, x_next_pred) * self.flux_loss_lambda
            # loss_prod_bhp_t1 += get_well_bhp_loss(x_next, x_next_pred, prod_loc) * self.bhp_loss_lambda

        loss_l2_reg = get_l2_reg_loss(z0)

        # loss_bound = loss_rec_t + loss_rec_t1/len(X_next) + loss_l2_reg + loss_flux_t + loss_flux_t1/len(X_next) + loss_prod_bhp_t + loss_prod_bhp_t1/len(X_next)
        loss_bound = loss_rec_t + loss_rec_t1 + loss_l2_reg + loss_flux_t+ loss_flux_t1
        # Use zt_logvar to approximate zt1_logvar_pred
        loss_trans = 0 
        for z_next, z_next_pred in zip(Z_next, Z_next_pred):
            loss_trans += get_l2_reg_loss(z_next- z_next_pred)
            
        loss_yobs = 0 
        for y_next, y_next_pred in zip(Yobs, Yobs_pred):
            loss_yobs += get_l2_reg_loss(y_next- y_next_pred)

        self.flux_loss = loss_flux_t + loss_flux_t1
        self.reconstruction_loss = loss_rec_t + loss_rec_t1
        self.well_loss = loss_prod_bhp_t + loss_prod_bhp_t1
        # self.total_loss = loss_bound + self.trans_loss_weight * loss_trans + g_A + g_B
        self.total_loss = loss_bound + self.trans_loss_weight * loss_trans + self.yobs_loss_weight* loss_yobs

        return self.total_loss

    def getFluxLoss(self):
        return self.flux_loss

    def getReconstructionLoss(self):
        return self.reconstruction_loss

    def getWellLoss(self):
        return self.well_loss

    def getTotalLoss(self):
        return self.total_loss


def get_reconstruction_loss(x, t_decoded):
    v = 0.1
#     return torch.mean(torch.sum((x.view(x.size(0), -1) - t_decoded.view(t_decoded.size(0), -1)) ** 2 / (2*v), dim=-1))
    return torch.mean(torch.sum((x.reshape(x.size(0), -1) - t_decoded.reshape(t_decoded.size(0), -1)) ** 2 / (2*v), dim=-1))


def get_l2_reg_loss(qm):
    l2_reg = 0.5 * qm.pow(2)
    return torch.mean(torch.sum(l2_reg, dim=-1))


def get_flux_loss(m, state, state_pred):
    perm = m
    p = state[:, -1,:, :].unsqueeze(1)
    p_pred = state_pred[:, -1, :, :].unsqueeze(1)

    tran_x = 1. / perm[:, :, 1:, ...] + 1. / perm[:, :, :-1, ...]
    tran_y = 1. / perm[:, :, :, 1:] + 1. / perm[:, :, :, :-1]
    flux_x = (p[:, :, 1:, ...] - p[:, :, :-1, ...]) / tran_x
    flux_y = (p[:, :, :, 1:] - p[:, :, :, :-1]) / tran_y
    flux_x_pred = (p_pred[:, :, 1:, ...] - p_pred[:, :, :-1, ...]) / tran_x
    flux_y_pred = (p_pred[:, :, :, 1:] - p_pred[:, :, :, :-1]) / tran_y

    loss_x = torch.sum(torch.abs(flux_x.reshape(flux_x.size(0), -1) - flux_x_pred.reshape(flux_x_pred.size(0), -1)), dim=-1)
    loss_y = torch.sum(torch.abs(flux_y.reshape(flux_y.size(0), -1) - flux_y_pred.reshape(flux_y_pred.size(0), -1)), dim=-1)

    loss_flux = torch.mean(loss_x + loss_y)
    return loss_flux


def get_binary_sat_loss(state, state_pred):
    sat_threshold = 0.105
    sat = state[:, :, :, 0].unsqueeze(-1)
    sat_pred = state_pred[:, :, :, 0].unsqueeze(-1)

    sat_bool = sat >= sat_threshold
    sat_bin = sat_bool.float()

    sat_pred_bool = sat_pred >= sat_threshold
    sat_pred_bin = sat_pred_bool.float()

    binary_loss = nn.functional.binary_cross_entropy(sat_pred_bin, sat_bin)
    return torch.mean(binary_loss)


def get_well_bhp_loss(state, state_pred, prod_well_loc):
    p_true = state[:, 1, prod_well_loc[:, 1], prod_well_loc[:, 0]].unsqueeze(1)
    p_pred = state_pred[:, 1, prod_well_loc[:, 1], prod_well_loc[:, 0]].unsqueeze(1)

    bhp_loss = torch.mean(torch.abs(p_true - p_pred))
    return bhp_loss
