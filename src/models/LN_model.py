import torch.nn.functional as F
from torch import nn, optim, utils, Tensor
import pytorch_lightning as pl
import wandb
import timm

class LN_model(pl.LightningModule):
    def __init__(self, model_name='resnet18', pretrained=True, in_chans=3):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, in_chans=in_chans)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        val_loss = self.criterion(preds, target)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)