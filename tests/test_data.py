import os
import pytest
import torch
from tests import _PATH_DATA
from src.data.LN_data_module import Flowers102DataModule
from torch.utils.data import DataLoader

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/data.pth"), reason="Data files not found")
def test_check_train_data():
    data = torch.load(_PATH_DATA + "/processed/processed_data.pth")
    train_set = data["train"]
    assert train_set != None

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/data.pth"), reason="Data files not found")
def test_check_test_data():
    data = torch.load(_PATH_DATA +"/processed/processed_data.pth")
    test_set = data["test"]
    assert test_set != None

@pytest.mark.skip(reason="Dataset not downloading in github action")
def test_Flowers102DataModule_train_data():
    data = Flowers102DataModule()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    assert len(train_dataloader) != 0, "First batch of the train dataloader has no data"
    assert isinstance(train_dataloader, DataLoader), "Train data not being converted to Dataloader object"

@pytest.mark.skip(reason="Dataset not downloading in github action")
def test_Flowers102DataModule_validation_data():
    data = Flowers102DataModule()
    data.setup("fit")
    val_dataloader = data.val_dataloader()
    assert len(val_dataloader) != 0, "First batch of the validation dataloader has no data"
    assert isinstance(val_dataloader, DataLoader), "Validation data not being converted to Dataloader object"

@pytest.mark.skip(reason="Dataset not downloading in github action")
def test_Flowers102DataModule_validation_sample_size():
    data = Flowers102DataModule()
    data.setup("fit")
    assert len(data.set_train) > len(data.set_val)*8, "Validation set is larger than expected (10%)"

@pytest.mark.skip(reason="Dataset not downloading in github action")
def test_Flowers102DataModule_test_data():
    data = Flowers102DataModule()
    data.setup("test")
    test_dataloader = data.test_dataloader()
    assert len(test_dataloader) != 0, "First batch of the test dataloader has no data"
    assert isinstance(test_dataloader, DataLoader), "Test data not being converted to Dataloader object"

@pytest.mark.skip(reason="Dataset not downloading in github action")
def test_Flowers102DataModule_predict_data():
    data = Flowers102DataModule()
    data.setup("predict")
    predict_dataloader = data.predict_dataloader()
    assert len(predict_dataloader) != 0, "First batch of the predict dataloader has no data"
    assert isinstance(predict_dataloader, DataLoader), "Predict data not being converted to Dataloader object"