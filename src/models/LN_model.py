from torch import nn, optim
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import timm
import wandb
import torch

class LN_model(pl.LightningModule):
    def __init__(self, model_name = 'resnet18', pretrained = True, 
                in_chans = 3, num_classes = 102, task = 'multiclass',
                optimizer = 'adam', lr = 1e-3, loss = 'CrossEntropyLoss',
                logger = 'wandb_loggger'):
        super().__init__()
        self.model = timm.create_model(
            model_name = model_name,
            pretrained = pretrained,
            in_chans = in_chans,
            num_classes = num_classes)

        self.num_classes = num_classes
        self.task = task
        self.wandb_logger = logger

        # optimizer parameters
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss() #getattr(nn,loss) 

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        if self.optimizer == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        # default optimizer
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        return preds, loss, acc

    def predict_step(self, batch, batch_idx):
        preds = self(batch)
        return preds.argmax(dim=-1)