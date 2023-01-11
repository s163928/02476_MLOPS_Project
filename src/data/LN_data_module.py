import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from glob import glob
import torch

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
    ])

class predictImageDataset(Dataset):
    def __init__(self, predict_dir):
        image_paths = glob(predict_dir + "//**.jpg") + glob(predict_dir + "//**.png")
        self.predict_images = torch.stack([transform(Image.open(x)) for x in image_paths])
    def __len__(self):
        return self.predict_images.size()[0]
    def __getitem__(self, idx):
        return self.predict_images[idx].float()

class Flowers102DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/raw", predict_dir: str = "./predict"):
        super().__init__()
        self.data_dir = data_dir
        self.predict_dir = predict_dir

    def prepare_data(self):
        # Download
        datasets.Flowers102(self.data_dir, "train", download=True)
        datasets.Flowers102(self.data_dir, "test", download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset_full = datasets.Flowers102(self.data_dir, "train", transform=transform)
            self.dataset_train, self.dataset_val = random_split(dataset_full, [0.9, 0.1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = datasets.Flowers102(self.data_dir, "test", transform=transform)

        # Assign predict dataset from folder when initialising class.
        if stage == "predict":
            self.dataset_predict = predictImageDataset(self.predict_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=68, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=68)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=68)

    # Placeholder...
    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=68)


