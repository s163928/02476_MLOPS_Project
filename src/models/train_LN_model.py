import pytorch_lightning as pl
import torchvision
from src.models.LN_model import LN_model
from src.data.LN_data_module import Flowers102DataModule
from torch.utils.data import DataLoader
import torchvision.transforms as tt

def main():
    print("Training day and night")

    model = LN_model()
    data = Flowers102DataModule()

    trainer = pl.Trainer(
        limit_train_batches=0.20, # Limit to 20% of total size.
        max_epochs=5,
        logger=pl.loggers.WandbLogger(project="flowers"),
        log_every_n_steps=1)
    trainer.fit(
        model=model,
        datamodule=data)


if __name__ == "__main__":
    main()