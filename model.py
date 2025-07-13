import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import ot
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *


class Model(object):
    def __init__(self, 
                 adata1,
                 adata2,
                 batch_size=500, 
                 training_steps=10000, 
                 seed=1234, 
                 n_latent=16, 
                 lambdaAE=10.0, 
                 lambdaLA=10.0, 
                 lambdaOT=1.0, 
                 lambdamGAN=1.0, 
                 lambdabGAN=1.0,
                 n_KNN=3,
                 mode='weak',
                 prior=False,
                 celltype_col=None,
                 source_col=None):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaAE = lambdaAE
        self.lambdaOT = lambdaOT
        self.lambdaLA = lambdaLA
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.n_KNN = n_KNN
        self.mode = mode
        self.prior = prior

        self.dataset_A = AnnDataDataset(adata1, celltype_key=celltype_col, source_key=source_col)
        self.dataset_B = AnnDataDataset(adata2, celltype_key=celltype_col, source_key=source_col)

        self.dataloader_A = load_data(self.dataset_A, self.batch_size)
        self.dataloader_B = load_data(self.dataset_B, self.batch_size)

    def _init_models_and_optimizers(self):
        self.E_A = encoder(self.dataset_A.feature_shapes['expression'], self.n_latent).to(self.device)
        self.E_B = encoder(self.dataset_B.feature_shapes['expression'], self.n_latent).to(self.device)
        self.G_A = generator(self.dataset_A.feature_shapes['expression'], self.n_latent).to(self.device)
        self.G_B = generator(self.dataset_B.feature_shapes['expression'], self.n_latent).to(self.device)

        self.D_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.D_A = MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device) \
            if self.dataset_A.source_categories > 1 else None
        self.D_B = MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device) \
            if self.dataset_B.source_categories > 1 else None

        self.params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + \
                        list(self.G_A.parameters()) + list(self.G_B.parameters())
        self.optimizer_G = optim.Adam(self.params_G, lr=0.001, weight_decay=0.001)

        self.params_D = list(self.D_Z.parameters())
        if self.D_A: self.params_D += list(self.D_A.parameters())
        if self.D_B: self.params_D += list(self.D_B.parameters())
        self.optimizer_D = optim.Adam(self.params_D, lr=0.001, weight_decay=0.001)

    def _set_train_mode(self):
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B]:
            if model is not None:
                model.train()

    def _set_eval_mode(self):
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B]:
            if model is not None:
                model.eval()

    def train(self):
        self._init_models_and_optimizers()
        self._set_train_mode()
        iterator_A, iterator_B = iter(self.dataloader_A), iter(self.dataloader_B)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            x_A, x_B = batch_A['expression'].float().to(self.device), batch_B['expression'].float().to(self.device)

            z_A, z_B = self.E_A(x_A), self.E_B(x_B)
            x_AtoB, x_BtoA = self.G_B(z_A), self.G_A(z_B)
            z_AtoB, z_BtoA = self.E_B(x_AtoB), self.E_A(x_BtoA)
            x_Arecon, x_Brecon = self.G_A(z_A), self.G_B(z_B)

            for _ in range(3):
                self.optimizer_D.zero_grad()
                loss_D_m = self.compute_discriminator_loss_inter(z_A, z_B)
                loss_D_b = self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)
                loss_D = self.lambdamGAN * loss_D_m + self.lambdabGAN * loss_D_b
                loss_D.backward()
                self.optimizer_D.step()

            loss_dict = {
                'AE': self.compute_ae_loss(x_A, x_Arecon) + self.compute_ae_loss(x_B, x_Brecon),
                'LA': self.compute_latent_align_loss(z_A, z_AtoB) + self.compute_latent_align_loss(z_B, z_BtoA),
                'OT': self.compute_ot_loss(z_A, z_B, batch_A, batch_B),
                'mGAN': self.compute_generator_loss_inter(z_A, z_B),
                'bGAN': self.compute_generator_loss_intra(z_A, z_B, batch_A, batch_B),
            }

            total_loss = (self.lambdaAE * loss_dict['AE'] +
                        self.lambdaLA * loss_dict['LA'] +
                        self.lambdaOT * loss_dict['OT'] +
                        self.lambdamGAN * loss_dict['mGAN'] + 
                        self.lambdabGAN * loss_dict['bGAN'])


            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 100 == 0:
                self.log(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")

    def eval(self):
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Evaluation started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['expression'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['expression'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)

        self.latent = np.concatenate((z_A.cpu().numpy(), z_B.cpu().numpy()), axis=0)

        end_time = time.time()
        print(f"Evaluation completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total evaluation time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A)+len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")

    def compute_ae_loss(self, x, x_recon):
        return F.mse_loss(x, x_recon)

    def compute_latent_align_loss(self, z, z_to):
        return F.mse_loss(z, z_to)

    def compute_discriminator_loss_inter(self, z_A, z_B):
        return F.softplus(-self.D_Z(z_A.detach())).mean() + F.softplus(self.D_Z(z_B.detach())).mean()

    def compute_discriminator_loss_intra(self, z_A, z_B, batch_A, batch_B):
        loss = 0.0
        if self.D_A:
            loss += F.cross_entropy(self.D_A(z_A.detach()), batch_A['source'].to(self.device))
        if self.D_B:
            loss += F.cross_entropy(self.D_B(z_B.detach()), batch_B['source'].to(self.device))
        return loss

    def compute_generator_loss_intra(self, z_A, z_B, batch_A, batch_B):
        loss = 0.0
        if self.D_A:
            loss += -F.cross_entropy(self.D_A(z_A), batch_A['source'].to(self.device))
        if self.D_B:
            loss += -F.cross_entropy(self.D_B(z_B), batch_B['source'].to(self.device))
        return loss

    def compute_generator_loss_inter(self, z_A, z_B):
        return -(F.softplus(-self.D_Z(z_A)) + F.softplus(self.D_Z(z_B))).mean()

    def compute_ot_loss(self, z_A, z_B, batch_A, batch_B):
        if 'link_feat' in batch_A and 'link_feat' in batch_B and self.mode == 'weak':
            c_cross = pairwise_correlation_distance(batch_A['link_feat'], batch_B['link_feat'])
        elif self.mode == 'strong':
            c_cross = pairwise_correlation_distance(z_A.detach().cpu(), z_B.detach().cpu())
        else:
            raise ValueError("Invalid mode for MNN computation")

        if 'celltype' in batch_A and 'celltype' in batch_B and self.prior:
            prior_matrix = build_celltype_prior(batch_A['celltype'], batch_B['celltype'], prior=2)
        else:
            prior_matrix = build_mnn_prior(c_cross, self.n_KNN, prior=2)

        p = ot.unif(z_A.size(0), type_as=c_cross)
        q = ot.unif(z_B.size(0), type_as=c_cross)
        plan = ot.partial.partial_wasserstein(p, q, c_cross * prior_matrix, m=0.5).to(self.device)

        z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
        return torch.sum(plan * z_dist) / torch.sum(plan)
    
    def log(self, step, loss_dict):
        print(f"Step {step} | "
            f"AE: {self.lambdaAE * loss_dict['AE']:.4f} | "
            f"LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"OT: {self.lambdaOT * loss_dict['OT']:.4f} | "
            f"mGAN: {self.lambdamGAN * loss_dict['mGAN']:.4f} | "
            f"bGAN: {self.lambdabGAN * loss_dict['bGAN']:.4f}")



