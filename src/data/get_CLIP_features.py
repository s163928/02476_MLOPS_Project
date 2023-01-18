from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import scipy.io
import torch
from glob import glob
import sys


class CLIPFeature(Dataset):
    def __init__(self, path_to_folder: str, label_file: str, model, processor) -> None:
        self.path_list = glob(path_to_folder + "jpg/*.jpg")
        print(len(self.path_list))
        self.model = model
        self.processor = processor
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


DATA_PATH = "data/raw/flowers-102/"
label_file = "imagelabels.mat"
BATCH_SIZE = 4
NUM_WORKERS = 4

# labels = []
# for i in [DATA_PATH + 'imagelabels.mat', DATA_PATH + 'imagelabels 2.mat']:
#     labels.extend(list(scipy.io.loadmat(i)['labels'][0]))


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # set either text=None or images=None when only the other is needed
# inputs = processor(images=image, return_tensors="pt", padding=True)

# img_features = model.get_image_features(inputs['pixel_values'])
# print(len(img_features.flatten().tolist()))

with open("./data/processed/features.csv", "w") as file:
    dataset = CLIPFeature(
        path_to_folder=DATA_PATH,
        label_file=label_file,
        model=model,
        processor=processor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        multiprocessing_context="fork",
    )
    for labels, features in dataloader:
        for i in range(len(labels)):
            print(features.shape)
            # print(labels[i].item(), ",".join(features), sep=",", file=file)
            sys.exit(1)
