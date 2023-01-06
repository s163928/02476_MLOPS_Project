import pytorch_lightning as pl
import torchvision
from src.models.LN_model import LN_model
from torch.utils.data import DataLoader
import torchvision.transforms as tt

transform = tt.Compose([
    tt.Resize((128,128)),
    tt.ToTensor()
])

def main():
    print("Training day and night")

    model = LN_model()

    train_set = torchvision.datasets.Flowers102('./data/raw', transform=transform)
    train_loader = DataLoader(train_set, batch_size=68, shuffle=True)

    trainer = pl.Trainer(
        limit_train_batches=0.20, # Limit to 20% of total size.
        max_epochs=5,
        logger=pl.loggers.WandbLogger(project="flowers"),
        log_every_n_steps=1)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader)


if __name__ == "__main__":
    main()