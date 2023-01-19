from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import scipy.io
import torch
from glob import glob

DATA_PATH = "data/raw/flowers-102/"
LABEL_FILE = "imagelabels.mat"
BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_FEATURES = 100  # 1-512


class CLIPFeature(Dataset):
    def __init__(self, path_to_folder: str, label_file: str) -> None:
        self.path_list = glob(path_to_folder + "jpg/*.jpg")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = list(scipy.io.loadmat(path_to_folder + label_file)["labels"][0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.path_list[index])
        label = self.labels[index]
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt", padding=True)
            img_features = self.model.get_image_features(inputs["pixel_values"])

        return label, img_features


def main():
    with open("./data/processed/features.csv", "w") as file:
        print(
            "target",
            ",".join(["feature_" + str(i) for i in range(NUM_FEATURES)]),
            sep=",",
            file=file,
        )

    dataset = CLIPFeature(
        path_to_folder=DATA_PATH,
        label_file=LABEL_FILE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        multiprocessing_context="fork",
    )

    with open("./data/processed/features.csv", "a") as file:
        for labels, features in dataloader:
            for i in range(len(labels)):
                feature_list = [
                    str(x) for x in features[i].view(-1).tolist()[:NUM_FEATURES]
                ]
                print(labels[i].item(), ",".join(feature_list), sep=",", file=file)


if __name__ == "__main__":
    main()
