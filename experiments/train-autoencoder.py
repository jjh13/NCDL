import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import random
from multiprocessing import freeze_support
from ncdl.modules.autoencoder import AE_Unet
import torchmetrics


from sklearn.metrics import f1_score
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderModule(pl.LightningModule):
    def __init__(self,
                 variant: str = 'unet_std',
                 learning_rate: float = 1e-3,
                 residual: bool = False,
                 loss: str = 'l1'):
        super().__init__()

        assert loss in ['l1', 'l2', 'bce']

        self.model = AE_Unet(3,
                             3,
                             variant,
                             residual=residual)
        self.loss = loss
        self.learning_rate = learning_rate
        self.bce_crit = nn.BCELoss(size_average=True)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        input = batch

        mask_pred = self.model(input)
        loss = F.l1_loss(mask_pred, input)
        self.log("val_l1_loss", F.l1_loss(mask_pred, input))
        self.log("val_l2_loss", F.mse_loss(mask_pred, input))
        self.log("val_ssim", torchmetrics.functional.structural_similarity_index_measure(preds=mask_pred, target=input))

        if batch_idx % 100 == 0:
            logger = self.logger.experiment
            grid = torchvision.utils.make_grid(torch.cat([
                input,mask_pred,
            ], dim=-2))

            logger.add_image(f"pred_images_{batch_idx}", grid, self.global_step)

    def training_step(self, batch, batch_idx):
        input = batch

        pred = self.model(input, iter=self.global_step)

        if self.loss == 'l2':
            loss = F.mse_loss(batch, pred)
        elif self.loss == 'l1':
            loss = F.l1_loss(batch, pred)

        self.log(f"loss_{self.loss}", loss.detach().item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        pass

if __name__ == '__main__':
    freeze_support()
    cli = LightningCLI(model_class=AutoencoderModule)


