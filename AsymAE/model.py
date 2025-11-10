import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from SymAE.dataloader import *
from SymAE.utils import *
from SymAE.network import *
from typing import Optional, Dict, Literal
from itertools import chain


class Model:
    def __init__(
        self,
        adata1,
        adata2,
        batch_size: int = 500,
        training_steps: int = 10000,
        seed: int = 1234,
        n_latent: int = 16,
        lambdaRecon_input: float = 10.0,
        lambdaRecon_output: float = 10.0,
        lambdaLA: float = 10.0,
        lambdaAlign: float = 1.0,
        lambdamGAN: float = 1.0,
        lambdabGAN: float = 1.0,
        lambdaGeo: float = 0.5,
        lambdaCLIP: float = 0.1,
        use_prior: bool = False,
        input_key = ['X_pca', 'X_lsi'],
        output_layer = [None, None],
        celltype_col: Optional[str] = None,
        source_col: Optional[str] = None,
        loss_type: Literal["MSE", "BCE"] = "MSE",
        cross_dist: Optional[np.ndarray] = None
    ) -> None:

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaRecon_input = lambdaRecon_input
        self.lambdaRecon_output = lambdaRecon_output
        self.lambdaAlign = lambdaAlign
        self.lambdaLA = lambdaLA
        self.lambdamGAN = lambdamGAN
        self.lambdabGAN = lambdabGAN
        self.lambdaGeo = lambdaGeo
        self.lambdaCLIP = lambdaCLIP
        self.use_prior = use_prior
        self.loss_type = loss_type
        
        self.celltype_col = celltype_col
        self.source_col = source_col

        self.dataset_A = AnnDataDataset(adata1, input_key = input_key[0], output_layer = output_layer[0], celltype_key=self.celltype_col, source_key=self.source_col)
        self.dataset_B = AnnDataDataset(adata2, input_key = input_key[1], output_layer = output_layer[1], celltype_key=self.celltype_col, source_key=self.source_col)

        self.dataloader_A_integration = load_data(self.dataset_A, self.batch_size)
        self.dataloader_B_integration = load_data(self.dataset_B, self.batch_size)

        self.dataloader_A_imputation = load_data(self.dataset_A, 16)
        self.dataloader_B_imputation = load_data(self.dataset_B, 16)

        self.cross_dist = torch.from_numpy(cross_dist).float()
        
        self._init_models_and_optimizers()

    def train_integration(self) -> None:
        self._set_train_integration_mode()
        iterator_A, iterator_B = iter(self.dataloader_A_integration), iter(self.dataloader_B_integration)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            input_A = batch_A['input'].float().to(self.device)
            input_B = batch_B['input'].float().to(self.device)

            z_A = self.E_A(input_A)
            z_B = self.E_B(input_B)

            input_AtoB = self.G_B(z_A)
            input_BtoA = self.G_A(z_B)

            z_AtoB = self.E_B(input_AtoB)
            z_BtoA = self.E_A(input_BtoA)

            input_Arecon = self.G_A(z_A)
            input_Brecon = self.G_B(z_B)

            loss_dict = {}

            # discriminator loss
            for _ in range(3):
                self.optimizer_D_m.zero_grad()
                loss_dict['mDis'] = self.compute_discriminator_loss_inter(z_A.detach(), z_B.detach())
                loss_dict['mDis'].backward()
                self.optimizer_D_m.step()

            if self.optimizer_D_b:
                self.optimizer_D_b.zero_grad()
                loss_dict['bDis'] = self.compute_discriminator_loss_intra(z_A.detach(), z_B.detach(), batch_A, batch_B)
                loss_dict['bDis'].backward()
                self.optimizer_D_b.step()
            else:
                loss_dict['bDis'] = torch.tensor(0.0, device=self.device)

            # input autoencoder loss
            loss_AE_A = torch.mean((input_Arecon - input_A)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(input_Arecon, input_A, reduction='mean')
            loss_AE_B = torch.mean((input_Brecon - input_B)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(input_Brecon, input_B, reduction='mean')
            loss_dict['AE_input'] = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_dict['LA'] = loss_LA_AtoB + loss_LA_BtoA
            
            # transport optimal process
            c_cross = self.cross_dist[batch_A["index"],:][:,batch_B["index"]].to(self.device)
            T = unbalanced_ot(cost_pp=c_cross, reg=0.05, reg_m=0.5, device=self.device)

            # modality align loss
            z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            loss_dict['Align'] = torch.sum(T * z_dist) / torch.sum(T)

            # geometric loss
            switch_interval = 100
            if step // switch_interval % 2 == 0:
                L_A = Graph_Laplacian_torch(z_A, nearest_neighbor=min(30, z_A.size(0)-1))
                z_A_new = Transform(z_A, z_B, T, L_A, lamda_Eigenvalue=0.5)
                loss_dict['Geo'] = torch.mean((z_A - z_A_new) ** 2)
            else:
                L_B = Graph_Laplacian_torch(z_B, nearest_neighbor=min(30, z_B.size(0)-1))
                z_B_new = Transform(z_B, z_A, torch.t(T), L_B, lamda_Eigenvalue=0.5)
                loss_dict['Geo'] = torch.mean((z_B - z_B_new) ** 2)
            
            # semi-supervised clip loss(optional)
            if self.use_prior:
                prior_matrix = build_celltype_prior(batch_A['celltype'], batch_B['celltype']).to(self.device)
                loss_dict['CLIP'] = generalized_clip_loss_stable_masked(z_A, z_B, prior_matrix)
            else:
                loss_dict['CLIP'] = torch.tensor(0.0, device=self.device)

            # generator loss
            loss_dict['mGAN'] = -self.compute_discriminator_loss_inter(z_A, z_B)
            loss_dict['bGAN'] = -self.compute_discriminator_loss_intra(z_A, z_B, batch_A, batch_B)
            
            total_loss = (
                self.lambdaRecon_input * loss_dict['AE_input']
                + self.lambdaLA * loss_dict['LA']
                + self.lambdaAlign * loss_dict['Align']
                + self.lambdaGeo * loss_dict['Geo']
                + self.lambdamGAN * loss_dict['mGAN']
                + self.lambdabGAN * loss_dict['bGAN']
                + self.lambdaCLIP * loss_dict['CLIP']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                self.log_integration(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")


    def train_imputation(self) -> None:
        self._set_train_imputation_mode()
        iterator_A, iterator_B = iter(self.dataloader_A_imputation), iter(self.dataloader_B_imputation)
        begin_time = time.time()
        print("Training started at:", time.asctime())

        for step in range(self.training_steps):
            batch_A, batch_B = next(iterator_A), next(iterator_B)
            input_A = batch_A['input'].float().to(self.device)
            input_B = batch_B['input'].float().to(self.device)
            output_A = batch_A['output'].float().to(self.device)
            output_B = batch_B['output'].float().to(self.device)

            z_A = self.E_A(input_A)
            z_B = self.E_B(input_B)

            input_AtoB = self.G_B(z_A)
            input_BtoA = self.G_A(z_B)

            z_AtoB = self.E_B(input_AtoB)
            z_BtoA = self.E_A(input_BtoA)

            input_Arecon = self.G_A(z_A)
            input_Brecon = self.G_B(z_B)

            output_Arecon = self.Decoder_A(input_Arecon)
            output_Brecon = self.Decoder_B(input_Brecon)

            loss_dict = {}

            # input autoencoder loss
            loss_AE_A = torch.mean((input_Arecon - input_A)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(input_Arecon, input_A, reduction='mean')
            loss_AE_B = torch.mean((input_Brecon - input_B)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(input_Brecon, input_B, reduction='mean')
            loss_dict['AE_input'] = loss_AE_A + loss_AE_B

            # output autoencoder loss
            loss_AE_A = torch.mean((output_Arecon - output_A)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(output_Arecon, output_A, reduction='mean')
            loss_AE_B = torch.mean((output_Brecon - output_B)**2) if self.loss_type == 'MSE' else F.binary_cross_entropy(output_Brecon, output_B, reduction='mean')
            loss_dict['AE_output'] = loss_AE_A + loss_AE_B

            # latent align loss
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_dict['LA'] = loss_LA_AtoB + loss_LA_BtoA
            
            total_loss = (
                self.lambdaRecon_input * loss_dict['AE_input']
                + self.lambdaRecon_output * loss_dict['AE_output']
                + self.lambdaLA * loss_dict['LA']
            )

            self.optimizer_G.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_G, 5.0)
            self.optimizer_G.step()

            if step % 1000 == 0:
                self.log_imputation(step, loss_dict)

        self.train_time = time.time() - begin_time
        print(f"Training finished. Time: {self.train_time:.2f} sec")


    def get_latent_representation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        input_A = torch.stack([self.dataset_A[i]['input'] for i in range(len(self.dataset_A))]).float().to(self.device)
        input_B = torch.stack([self.dataset_B[i]['input'] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            z_A = self.E_A(input_A)
            z_B = self.E_B(input_B)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(input_A) + len(input_B)} samples")
        print(f"Latent space shape: {self.latent.shape}")

    def get_imputation(self) -> None:
        self._set_eval_mode()
        begin_time = time.time()
        print(f"Started at: {time.asctime(time.localtime(begin_time))}")

        input_A = torch.stack([self.dataset_A[i]["input"] for i in range(len(self.dataset_A))]).float().to(self.device)
        input_B = torch.stack([self.dataset_B[i]["input"] for i in range(len(self.dataset_B))]).float().to(self.device)

        with torch.no_grad():
            z_A = self.E_A(input_A)
            z_B = self.E_B(input_B)
            input_AtoB = self.G_B(z_A)
            input_BtoA = self.G_A(z_B)

            if getattr(self.dataset_A, "output", None) is not None and self.Decoder_A is not None:
                self.imputed_BtoA = self.Decoder_A(input_BtoA).detach().cpu().numpy()
            else:
                self.imputed_BtoA = input_BtoA.detach().cpu().numpy()

            if getattr(self.dataset_B, "output", None) is not None and self.Decoder_B is not None:
                self.imputed_AtoB = self.Decoder_B(input_AtoB).detach().cpu().numpy()
            else:
                self.imputed_AtoB = input_AtoB.detach().cpu().numpy()

        end_time = time.time()
        print(f"Completed at: {time.asctime(time.localtime(end_time))}")
        print(f"Total time: {end_time - begin_time:.2f} seconds")
        print(f"Processed {len(input_A) + len(input_B)} samples")

    def compute_discriminator_loss_inter(self, z_A: torch.Tensor, z_B: torch.Tensor) -> torch.Tensor:
        return F.softplus(-self.D_Z(z_A)).mean() + F.softplus(self.D_Z(z_B)).mean()

    def compute_discriminator_loss_intra(self, z_A, z_B, batch_A, batch_B):
        losses = []
        if self.D_A:
            losses.append(F.cross_entropy(self.D_A(z_A), batch_A['source'].to(self.device)))
        if self.D_B:
            losses.append(F.cross_entropy(self.D_B(z_B), batch_B['source'].to(self.device)))
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return sum(losses)

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
            n_output=self.dataset_A.feature_shapes['input'],
            loss_type=self.loss_type
        ).to(self.device)

        self.G_B = Generator(
            n_latent=self.n_latent,
            n_output=self.dataset_B.feature_shapes['input'],
            loss_type=self.loss_type
        ).to(self.device)

        if getattr(self.dataset_A, "output", None) is not None:
            self.Decoder_A = Generator(
                n_latent=self.dataset_A.feature_shapes['input'],
                n_output=self.dataset_A.feature_shapes['output'],
                loss_type=self.loss_type
            ).to(self.device)
        else:
            self.Decoder_A = None

        if getattr(self.dataset_B, "output", None) is not None:
            self.Decoder_B = Generator(
                n_latent=self.dataset_B.feature_shapes['input'],
                n_output=self.dataset_B.feature_shapes['output'],
                loss_type=self.loss_type
            ).to(self.device)
        else:
            self.Decoder_B = None

        modules = [self.E_A, self.E_B, self.G_A, self.G_B]
        if self.Decoder_A is not None:
            modules.append(self.Decoder_A)
        if self.Decoder_B is not None:
            modules.append(self.Decoder_B)

        self.params_G = chain.from_iterable(m.parameters() for m in modules)

        self.optimizer_G = optim.AdamW(self.params_G, lr=1e-3, weight_decay=1e-3)
        self.D_Z = BinaryDiscriminator(self.n_latent).to(self.device)
        self.optimizer_D_m = optim.AdamW(self.D_Z.parameters(), lr=1e-3, weight_decay=1e-3)

        self.D_A: Optional[nn.Module] = (
            MultiClassDiscriminator(self.n_latent, self.dataset_A.source_categories).to(self.device)
            if self.dataset_A.source_categories > 1 else None
        )
        self.D_B: Optional[nn.Module] = (
            MultiClassDiscriminator(self.n_latent, self.dataset_B.source_categories).to(self.device)
            if self.dataset_B.source_categories > 1 else None
        )

        params_D_b = []
        if self.D_A:
            params_D_b.extend(self.D_A.parameters())
        if self.D_B:
            params_D_b.extend(self.D_B.parameters())
        self.optimizer_D_b = optim.AdamW(params_D_b, lr=1e-3, weight_decay=1e-3) if params_D_b else None


    def _set_train_integration_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B]:
            if model is not None:
                model.train()

    def _set_train_imputation_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.Decoder_A, self.Decoder_B]:
            if model is not None:
                model.train()
        
        for encoder in [self.E_A, self.E_B]:
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False

    def _set_eval_mode(self) -> None:
        for model in [self.E_A, self.E_B, self.G_A, self.G_B, self.D_Z, self.D_A, self.D_B, self.Decoder_A, self.Decoder_B]:
            if model is not None:
                model.eval()

    def log_integration(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        print(
            f"Step {step} | "
            f"loss_AE_input: {self.lambdaRecon_input * loss_dict['AE_input']:.4f} | "
            f"loss_LA: {self.lambdaLA * loss_dict['LA']:.4f} | "
            f"loss_Align: {self.lambdaAlign * loss_dict['Align']:.4f} | "
            f"loss_Geo: {self.lambdaGeo * loss_dict['Geo']:.4f} | "
            f"loss_CLIP: {self.lambdaCLIP * loss_dict['CLIP']:.4f} | "
            f"loss_mGAN vs loss_mDis: {loss_dict['mGAN']:.4f} vs {loss_dict['mDis']:.4f} | "
            f"loss_bGAN vs loss_bDis: {loss_dict['bGAN']:.4f} vs {loss_dict['bDis']:.4f}"
        )

    def log_imputation(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        print(
            f"Step {step} | "
            f"loss_AE_input: {self.lambdaRecon_input * loss_dict['AE_input']:.4f} | "
            f"loss_AE_output: {self.lambdaRecon_output * loss_dict['AE_output']:.4f} | "
            f"loss_LA: {self.lambdaLA * loss_dict['LA']:.4f} "
        )



