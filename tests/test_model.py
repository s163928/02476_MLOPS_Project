import pytest
import torch
from src.models.LN_model import LN_model
from torch import nn

def test_LN_model_model():
    test_model = LN_model()
    assert test_model.model != None, "LN_model has no model defined"
    assert isinstance(test_model.model, nn.Module), "The model from LN_model does not have the 'nn.Module' type"

def test_LN_model_criterion():
    test_model = LN_model()
    assert is_torch_loss(test_model.criterion), "The model from LN_model does not have the 'nn.Module' type"
    assert test_model.criterion != None, "LN_model has no criterion defined"

def test_LN_model_forward():
    test_model = LN_model()
    x = torch.randn(1, 3, 224, 224)
    test_model.model(x)

def test_LN_model_optimizer():
    test_model = LN_model()
    optimizer = test_model.configure_optimizers()
    assert optimizer != None, "LN_model has no optimizer defined"
    assert is_torch_optim(optimizer), "The model from LN_model does not have the 'nn.Module' type"


def is_torch_loss(criterion) -> bool:
    type_ = str(type(criterion)).split("'")[1]
    parent = type_.rsplit(".", 1)[0]
    return parent == "torch.nn.modules.loss"

def is_torch_optim(optimizer) -> bool:
    type_ = str(type(optimizer)).split("'")[1]
    return "torch.optim" in type_