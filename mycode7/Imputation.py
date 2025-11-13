import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mycode.dataloader import *
from mycode.utils import *
from mycode.network import *
from typing import Optional, Dict, Literal
from itertools import chain, cycle
import os


class ImputationModel:
    def __init__(
        self,
        adata_A: "anndata.AnnData",
        adata_B: "anndata.AnnData",
        input_key: List[Optional[str]] = ['X_pca', 'X_lsi'],
        output_layer: List[Optional[str]] = ['counts', 'counts'],
        batch_size: int = 16,
        training_steps: int = 10000,
        seed: int = 1234,
        n_latent: int = 10,
        lambdaRecon_x: float = 10.0,
        lambdaRecon_y: float = 10.0,
        lambdaLA: float = 10.0,
        model_path = "models"
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaRecon_x = lambdaRecon_x
        self.lambdaRecon_y = lambdaRecon_y
        self.lambdaLA = lambdaLA

        self.dataset_A = AnnDataDataset(
            adata_A,
            input_key = input_key[0],
            output_layer = output_layer[0],
            mode="imputation"
        )
        self.dataset_B = AnnDataDataset(
            adata_B,
            input_key = input_key[1],
            output_layer = output_layer[1],
            mode="imputation"
        )
        
        self.dataloader_A = load_data(self.dataset_A, batch_size = self.batch_size, mode = "imputation")
        self.dataloader_B = load_data(self.dataset_B, batch_size = self.batch_size, mode = "imputation")

        self.model_path = model_path
        

    def train(self) -> None:
        self._init_models_and_optimizers()
        self._set_train_mode()
        iterator_A = cycle(self.dataloader_A)
        iterator_B = cycle(self.dataloader_B)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            x_A = batch_A['input'].float().to(self.device)
            x_B = batch_B['input'].float().to(self.device)
            y_A = batch_A['output'].float().to(self.device)
            y_B = batch_B['output'].float().to(self.device)
            library_size_A = y_A.sum(dim=1, keepdim=True)
            library_size_B = y_B.sum(dim=1, keepdim=True)

            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)

            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)

            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)

            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)

            loss_dict = {}

            # x reconstruction loss
            loss_reconx_A = torch.mean((x_Arecon - x_A)**2) 
            loss_reconx_B = torch.mean((x_Brecon - x_B)**2) 
            loss_dict['Recon_x'] = loss_reconx_A + loss_reconx_B

            # latent align loss
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_dict['LA'] = loss_LA_AtoB + loss_LA_BtoA

            # y reconstruction loss            
            # pyA_scale, pyA_r = self.D_A(x_Arecon)
            # pyB_scale, pyB_r = self.D_B(x_Brecon)
            # pyA_rate = pyA_scale * library_size_A
            # pyB_rate = pyB_scale * library_size_B
            # loss_recony_A = -log_nb_positive(y_A, pyA_rate, pyA_r).mean() / y_A.shape[1]
            # loss_recony_B = -log_nb_positive(y_B, pyB_rate, pyB_r).mean() / y_B.shape[1]
            # loss_dict['Recon_y'] = loss_recony_A + loss_recony_B
            y_Arecon = self.D_A(x_Arecon)
            y_Brecon = self.D_B(x_Brecon)
            loss_recony_A = torch.mean((y_Arecon - y_A)**2) 
            loss_recony_B = torch.mean((y_Brecon - y_B)**2) 
            loss_dict['Recon_y'] = loss_recony_A + loss_recony_B

            total_loss = (
                self.lambdaRecon_x * loss_dict['Recon_x']
                + self.lambdaLA * loss_dict['LA']
                + self.lambdaRecon_y * loss_dict['Recon_y']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                self.log(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")
        

    def get_latent_representation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")


    def get_imputation(self, library_size=10000) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        x_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        x_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            # pyAtoB_scale, _ = self.D_B(x_AtoB)
            # pyBtoA_scale, _ = self.D_A(x_BtoA)
            # pyAtoB_rate = pyAtoB_scale * library_size
            # pyBtoA_rate = pyBtoA_scale * library_size 
            # self.imputed_AtoB = pyAtoB_rate.detach().cpu().numpy()
            # self.imputed_BtoA = pyBtoA_rate.detach().cpu().numpy()
            y_AtoB = self.D_B(x_AtoB)
            y_BtoA = self.D_A(x_BtoA)
            self.imputed_AtoB = y_AtoB.detach().cpu().numpy()
            self.imputed_BtoA = y_BtoA.detach().cpu().numpy()

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(x_A) + len(x_B)} samples")


    def _init_models_and_optimizers(self) -> None:
        self.E_A = Encoder(
            n_input=self.dataset_A.feature_shapes['input'],
            n_latent=self.n_latent
        ).to(self.device)

        self.E_B = Encoder(
            n_input=self.dataset_B.feature_shapes['input'],
            n_latent=self.n_latent
        ).to(self.device)

        self.G_A = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_A.feature_shapes['input']
        ).to(self.device)

        self.G_B = Generator(
            n_latent=self.n_latent,
            n_input=self.dataset_B.feature_shapes['input']
        ).to(self.device)

        self.D_A = Decoder(
            n_input=self.dataset_A.feature_shapes['input'],
            n_output=self.dataset_A.feature_shapes['output']
        ).to(self.device)

        self.D_B = Decoder(
            n_input=self.dataset_B.feature_shapes['input'],
            n_output=self.dataset_B.feature_shapes['output']
        ).to(self.device)

        ckpt_path = os.path.join(self.model_path, "ckpt.pth")
        if self.model_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.E_A.load_state_dict(checkpoint['E_A'])
            self.E_B.load_state_dict(checkpoint['E_B'])
            self.G_A.load_state_dict(checkpoint['G_A'])
            self.G_B.load_state_dict(checkpoint['G_B'])

            print(f"✅ Loaded checkpoint from {ckpt_path}")
        else:
            print("⚠️ No checkpoint found, training from scratch.")

        for encoder in [self.E_A, self.E_B]:
            for param in encoder.parameters():
                param.requires_grad = False

        self.params_G = chain(
            self.G_A.parameters(),
            self.G_B.parameters(),
            self.D_A.parameters(),
            self.D_B.parameters()
        )

        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=1e-3)

    def _set_train_mode(self) -> None:
        for model in [self.G_A, self.G_B, self.D_A, self.D_B]:
            if model is not None:
                model.train()

    def _set_eval_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_A, self.D_B]:
            if model is not None:
                model.eval()

    def log(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        print(
            f"Step {step} | "
            f"loss_Recon_x: {self.lambdaRecon_x * loss_dict['Recon_x']:.4f} | "
            f"loss_LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"loss_Recon_y: {self.lambdaRecon_y * loss_dict['Recon_y']:.4f} "
        )



