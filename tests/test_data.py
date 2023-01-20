from torch.utils.data import DataLoader

from src.data.LN_data_module import Flowers102DataModule
from src.data.get_CLIP_features import CLIPFeature


def test_Flowers102DataModule_train_data():
    data = Flowers102DataModule()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    assert len(train_dataloader) != 0, "First batch of the train dataloader has no data"
    assert isinstance(
        train_dataloader, DataLoader
    ), "Train data not being converted to Dataloader object"


def test_Flowers102DataModule_validation_data():
    data = Flowers102DataModule()
    data.setup("fit")
    val_dataloader = data.val_dataloader()
    assert (
        len(val_dataloader) != 0
    ), "First batch of the validation dataloader has no data"
    assert isinstance(
        val_dataloader, DataLoader
    ), "Validation data not being converted to Dataloader object"


def test_Flowers102DataModule_validation_sample_size():
    data = Flowers102DataModule()
    data.setup("fit")
    assert (
        len(data.dataset_train) > len(data.dataset_val) * 8
    ), "Validation set is larger than expected (10%)"


def test_Flowers102DataModule_test_data():
    data = Flowers102DataModule()
    data.setup("test")
    test_dataloader = data.test_dataloader()
    assert len(test_dataloader) != 0, "First batch of the test dataloader has no data"
    assert isinstance(
        test_dataloader, DataLoader
    ), "Test data not being converted to Dataloader object"


def test_Flowers102DataModule_predict_data():
    example_image = "data/raw/flowers-102/jpg/image_00001.jpg"
    data = Flowers102DataModule(predict_data=example_image)
    data.setup("predict")
    predict_dataloader = data.predict_dataloader()
    assert (
        len(predict_dataloader) != 0
    ), "First batch of the predict dataloader has no data"
    assert isinstance(
        predict_dataloader, DataLoader
    ), "Predict data not being converted to Dataloader object"


def test_CLIPFeature_get_item():
    DATA_PATH = "data/raw/flowers-102/"
    LABEL_FILE = "imagelabels.mat"
    dataset = CLIPFeature(DATA_PATH, LABEL_FILE)
    feature_dataloader = DataLoader(dataset)
    _, feature = next(iter(feature_dataloader))
    assert (
        len(feature_dataloader) != 0
    ), "First batch of the CLIP feature dataloader has no data"
    assert feature.shape == (1, 1, 512), "Feature data is not correct shape"
