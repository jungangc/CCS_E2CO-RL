import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from MSE2C import MSE2C, MSE2CNODE
from MSloss import CustomizedLoss

class ROMWithE2C(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prob, num_inj, input_shape, perm_shape, prod_loc_shape, learning_rate, method, ode_steps,
                 n_steps, sigma=0.0, lambda_flux_loss=1/1000., lambda_bhp_loss=20.0, lambda_trans_loss=1.0, lambda_yobs_loss = 1.0):
        super(ROMWithE2C, self).__init__()
        if method == 'E2C':
            self.model = MSE2C(latent_dim, u_dim, num_prob, num_inj, input_shape, perm_shape, prod_loc_shape, sigma, n_steps)
        elif method == 'E2CNODE':
            self.model = MSE2CNODE(latent_dim, u_dim, input_shape, perm_shape, prod_loc_shape, ode_steps, sigma)
        else:
            print('wrong methods chosen')
        self.loss_object = CustomizedLoss(lambda_flux_loss, lambda_bhp_loss, lambda_trans_loss, lambda_yobs_loss)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_loss = torch.tensor(0.0)
        self.train_reconstruction_loss = torch.tensor(0.0)
        self.train_flux_loss = torch.tensor(0.0)
        self.train_well_loss = torch.tensor(0.0)
        self.test_loss = torch.tensor(0.0)

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            # xt, ut, yt, dt, perm = inputs
            xt, ut, yt, dt = inputs
            xt1_pred, yt1_pred = self.model.predict(inputs)        
        return xt1_pred, yt1_pred
    
    def predict_latent(self, zt, dt, ut):
        self.model.eval()
        with torch.no_grad():
            zt_next, yt_next = self.model.predict_latent(zt, dt, ut)  

        return zt_next, yt_next

    def evaluate(self, inputs):
        self.model.eval()
        with torch.no_grad():
            # xt1 = labels.float()
            # xt, ut, dt, _, _ = inputs
            
            predictions = self.model(inputs)
            # Parse predictions
            # X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec, perm = predictions
            X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec = predictions
            
            # zt1 = self.model.encoder(xt1)
            # y_pred = (xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc)
            t_loss = self.loss_object(predictions)

            self.test_loss = t_loss

    def update(self, inputs):
        self.model.train()

        # xt1=labels
        # X, U, Y, dt, perm = inputs
        X, U, Y, dt = inputs
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        # Parse predictions
        # X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec, perm = predictions
        X_next_pred, X_next, Z_next_pred, Z_next, Y_next_pred, Y, z0, x0, x0_rec = predictions
        # zt1 = self.model.encoder(xt1)
        # y_pred = (xt1_pred, zt1_pred, zt1, zt, xt_rec, xt, perm, prod_loc)
        loss = self.loss_object(predictions)

        loss.backward()
        self.optimizer.step()

        self.train_loss = loss
        self.train_flux_loss = self.loss_object.getFluxLoss()
        self.train_reconstruction_loss = self.loss_object.getReconstructionLoss()
        self.train_well_loss = self.loss_object.getWellLoss()

    def get_train_loss(self):
        return self.train_loss.item()

    def get_train_flux_loss(self):
        return self.train_flux_loss.item()

    def get_train_reconstruction_loss(self):
        return self.train_reconstruction_loss.item()

    def get_train_well_loss(self):
        return self.train_well_loss.item()

    def get_test_loss(self):
        return self.test_loss.item()