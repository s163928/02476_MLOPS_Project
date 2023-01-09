import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision import transforms


class Flowers102DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/raw"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
            ])

    def prepare_data(self):
        # Download
        datasets.Flowers102(self.data_dir, "train", download=True)
        datasets.Flowers102(self.data_dir, "test", download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            set_full = datasets.Flowers102(self.data_dir, "train", transform=self.transform)
            self.set_train, self.set_val = random_split(set_full, [0.9, 0.1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.set_test = datasets.Flowers102(self.data_dir, "test", transform=self.transform)

        # Placeholder, prediction data needs to be defined here...
        if stage == "predict":
            self.set_predict = datasets.Flowers102(self.data_dir, "test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.set_train, batch_size=68, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.set_val, batch_size=68)

    def test_dataloader(self):
        return DataLoader(self.set_test, batch_size=68)

    # Placeholder...
    def predict_dataloader(self):
        return DataLoader(self.set_predict, batch_size=68)