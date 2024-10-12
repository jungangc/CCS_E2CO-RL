# from layers import *

import torch
import torch.nn as nn

from MSE2C_layers import *

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)
        
class LinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prob, num_inj):
        super(LinearTransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim 
        self.num_prob = num_prob 
        self.num_inj = num_inj
        self.trans_encoder = create_trans_encoder(self.latent_dim + 1)
        self.trans_encoder.apply(weights_init)
        
        self.At_layer = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(self.latent_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        
        self.Ct_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj)* self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj) * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)

        Zt_k =[]
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            yt = torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            Zt_k.append(zt)
            Yt_k.append(yt)
        # print('predicted At shape', At.shape)
        # return Zt_k, gershgorin_loss(At), gershgorin_loss(Bt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)
        
        ut_dt = ut * dt

        zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
#         print('predicted latent shape', zt1.shape)
        return zt1, yt1

class LinearMultiTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prob, num_inj, nsteps):
        super(LinearMultiTransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim 
        self.num_prob = num_prob 
        self.num_inj = num_inj
        self.nsteps = nsteps
        self.trans_encoder = create_trans_encoder(self.latent_dim + 1)
        self.trans_encoder.apply(weights_init)
        
        self.At_layer = nn.Linear(self.latent_dim, self.latent_dim * self.latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(self.latent_dim, self.latent_dim * self.u_dim)
        self.Bt_layer.apply(weights_init)
        
        self.Ct_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj)* self.latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(self.latent_dim, (self.num_prob*2+ self.num_inj) * self.u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)

        Zt_k =[]
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            yt = torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
            Zt_k.append(zt)
            Yt_k.append(yt)
        # print('predicted At shape', At.shape)
        # return Zt_k, gershgorin_loss(At), gershgorin_loss(Bt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
#         print(dt.shape)
#         print(zt.shape)
        zt_expand = torch.cat([zt, dt], dim=-1)

        hz = self.trans_encoder(zt_expand)
#         print(hz.shape)
        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        Ct = self.Ct_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.num_prob*2+ self.num_inj, self.u_dim)
        
        ut_dt = ut * dt

        zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        yt1 = torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1) + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1)
            
#         print('predicted latent shape', zt1.shape)
        return zt1, yt1

class NonLinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, nsteps):
        super(NonLinearTransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.latent_dim_total = latent_dim + u_dim
        self.u_dim = u_dim 
        self.node_encoder = create_node_encoder(self.latent_dim_total, self.u_dim)
        self.steps = nsteps
        # self.feature = ode
        self.norm = nn.BatchNorm2d(128)

    def forward(self, zt, dt, ut):
        hz = self.node_encoder
        # ut_dt = ut * dt
        zt1 = ode_solve(zt, ut, dt, self.steps, hz)
        # zt1 = torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1)
        
#         print('predicted latent shape', zt1.shape)
        return zt1

def ode_solve(z0, ut, dt, nsteps, func):
    n_steps = nsteps
    z = z0
    for i_step in range(n_steps):
        z = z + dt/n_steps * func(z, ut)
        # t = t + dt
    return z
    

def create_trans_encoder(total_input_dim):
    trans_encoder = nn.Sequential(
        fc_bn_relu(total_input_dim, 200),
        fc_bn_relu(200, 200),
        fc_bn_relu(200, total_input_dim - 1)
    )
    return trans_encoder

def create_nltrans_encoder(total_input_dim, u_dim):
    trans_encoder = nn.Sequential(
        fc_bn_relu(total_input_dim, 200),
        fc_bn_relu(200, 200),
        fc_bn_relu(200, total_input_dim - u_dim)
    )
    return trans_encoder

class create_node_encoder(nn.Module):
    def __init__(self, total_input_dim, u_dim):
        super(create_node_encoder, self).__init__()
        # self.input_dim = input_dim
        self.NNODE = create_nltrans_encoder(total_input_dim, u_dim)
    def forward(self, x, u):
        zt_expand = torch.cat([x, u], dim=-1)
        out = self.NNODE(zt_expand)
        return out
        

def fc_bn_relu(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape, sigma=0.0):
        super(Encoder, self).__init__()

        self.sigma = sigma
        self.fc_layers = nn.Sequential(
            conv_bn_relu(input_shape[0], 16, 3, 3, stride=2),
            conv_bn_relu(16, 32, 3, 3, stride=1),
            conv_bn_relu(32, 64, 3, 3, stride=2),
            conv_bn_relu(64, 128, 3, 3, stride=1)
        )
        self.fc_layers.apply(weights_init)
        
        self.res_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1)
        )
        self.res_layers.apply(weights_init)

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(128 * int(input_shape[1] / 4) * int(input_shape[2] / 4), latent_dim)

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.res_layers(x)
        x = self.flatten(x)
        xi_mean = self.fc_mean(x)
        return xi_mean

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Decoder, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, int(input_shape[1] * input_shape[2] / 16 * 128)),
            nn.ReLU()
        )
        self.fc_layers.apply(weights_init)
        
        self.upsample_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3)
        )
        self.upsample_layers.apply(weights_init)

        self.deconv_layers = nn.Sequential(
#             dconv_bn_nolinear(128, 64, 3, 3, stride=(1, 1)),
#             dconv_bn_nolinear(64, 32, 3, 3, stride=(2, 2)),
#             dconv_bn_nolinear(32, 16, 3, 3, stride=(1, 1)),
#             dconv_bn_nolinear(16, input_shape[0], 3, 3, stride=(2, 2)),
            dconv_bn_nolinear(128, 64, 3, 3, stride=(1, 1), padding =1),
            dconv_bn_nolinear(64, 32, 2, 2, stride=(2, 2)),
            dconv_bn_nolinear(32, 16, 3, 3, stride=(1, 1), padding =1),
            dconv_bn_nolinear(16, 16, 2, 2, stride=(2, 2)),
            nn.Conv2d(16, input_shape[0], kernel_size=(3, 3), padding='same')
        )
        self.deconv_layers.apply(weights_init)
        
        self.input_shape=input_shape

    def forward(self, z):
       
        x = self.fc_layers(z)
#         print(self.input_shape)
        x = x.view(-1, 128, int(self.input_shape[1] / 4), int(self.input_shape[2] / 4))
        x = self.upsample_layers(x)
        y = self.deconv_layers(x)
        return y

class MSE2C(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prob, num_inj, input_shape, perm_shape, prod_loc_shape, n_steps, sigma=0.0):
        super(MSE2C, self).__init__()
        self._build_model(latent_dim, u_dim, num_prob, num_inj, input_shape, sigma)
        self.perm_shape = perm_shape
        self.prod_loc_shape = prod_loc_shape
        self.n_steps = n_steps

    def _build_model(self, latent_dim, u_dim, num_prob, num_inj, input_shape, sigma):
        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = LinearTransitionModel(latent_dim, u_dim, num_prob, num_inj)

    def forward(self, inputs):
        
        # X, U, Y, dt, perm = inputs
        X, U, Y, dt = inputs

        X_next = X[1:]
        x0 = X[0]
        z0 = self.encoder(x0)
        x0_rec = self.decoder(z0)
        
        X_next_pred = []
        Z_next = []
        # Z_next_pred = []

        # q_z = q_z0
        z = z0
        # Z_next_pred, g_A, g_B = self.transition.forward_nsteps(z, dt, U)
        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z, dt, U)
        # print(self.n_steps)
        for i_step in range(len(Z_next_pred)):
            z_next = Z_next_pred[i_step]

            x_next_pred = self.decoder(z_next)

            z_next = self.encoder(X[i_step + 1])
            
            X_next_pred.append(x_next_pred)
            Z_next.append(z_next)
            
        # self.zt1_pred = self.transition(self.zt, self.dt, self.ut)
        # self.xt1_pred = self.decoder(self.zt1_pred)
        # return X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec, perm
        return X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec
    
    def predict(self, inputs):
        
        # xt, ut, yt, dt, perm = inputs
        xt, ut, yt, dt = inputs
        # assert self.n_steps == len(U) == len(X) - 1

        zt = self.encoder(xt)
        zt_next, yt_next = self.transition(zt, dt, ut)
        xt_next_pred = self.decoder(zt_next)
            
        # self.zt1_pred = self.transition(self.zt, self.dt, self.ut)
        # self.xt1_pred = self.decoder(self.zt1_pred)
        # return X_next_pred, X_next, Z_next_pred, Z_next, z0, x0, x0_rec, g_A, g_B, perm, prod_loc
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):

        zt_next, yt_next = self.transition(zt, dt, ut)

        return zt_next, yt_next

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))
        self.transition.load_state_dict(torch.load(transition_file))
        
    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        torch.save(self.encoder.state_dict(), encoder_file)
        torch.save(self.decoder.state_dict(), decoder_file)
        torch.save(self.transition.state_dict(), transition_file)

        
class MSE2CNODE(nn.Module):
    def __init__(self, latent_dim, u_dim, input_shape, perm_shape, prod_loc_shape, ode_steps, sigma=0.0):
        super(MSE2CNODE, self).__init__()
        self._build_model(latent_dim, u_dim, input_shape, sigma, ode_steps)
        self.perm_shape = perm_shape
        self.prod_loc_shape = prod_loc_shape
        self.steps = ode_steps

    def _build_model(self, latent_dim, u_dim, input_shape, sigma, nsteps):
        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = NonLinearTransitionModel(latent_dim, u_dim, nsteps)

    def forward(self, inputs):
        self.xt, self.ut, self.dt, self.perm, self.prod_loc = inputs
        # nsteps = self.steps
        self.zt = self.encoder(self.xt)
        self.xt_rec = self.decoder(self.zt)
        self.zt1_pred = self.transition(self.zt, self.dt, self.ut)
        self.xt1_pred = self.decoder(self.zt1_pred)
        return self.xt1_pred, self.zt1_pred, self.zt, self.xt_rec, self.perm, self.prod_loc

    def load_weights_from_file(self, encoder_file, decoder_file, transition_file):
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))
        self.transition.load_state_dict(torch.load(transition_file))
        
    def save_weights_to_file(self, encoder_file, decoder_file, transition_file):
        torch.save(self.encoder.state_dict(), encoder_file)
        torch.save(self.decoder.state_dict(), decoder_file)
        torch.save(self.transition.state_dict(), transition_file)