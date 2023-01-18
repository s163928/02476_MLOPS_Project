import torch.nn.functional as F
from torch import nn, optim, utils, Tensor
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import wandb
import timm

class LN_model(pl.LightningModule):
    def __init__(self, model_name='resnet18', pretrained=True, 
    in_chans=3, num_classes=102,lr=1e-3, loss='CrossEntropyLoss'):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes)

        # optimizer parameters
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss() #getattr(nn,loss) 

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = accuracy(preds, target)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        val_loss = self.criterion(preds, target)
        val_acc = accuracy(preds, target)
        self.log("val_loss", val_loss)
        self.log("train_accuracy", val_acc)

    def predict_step(self, batch, batch_idx):
        preds = self(batch)
        return preds.argmax(dim=-1)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)