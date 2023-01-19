import click
import os
import pytorch_lightning as pl
from src.data.LN_data_module import Flowers102DataModule
from src.models.LN_model import LN_model

# Takes a model.ckpt and an image or folder with images and returns predicted classes.


@click.command()
@click.option(
    "--model",
    default="models/model.ckpt",
    help="Model checkpoint for prediction",
)
@click.option("--data", default="./predict", help="learning rate to use for training")
def _predict(model, data):
    predict(model=model, data=data)


def predict(data, model: str = "models/model.ckpt"):
    if os.path.exists(model):
        model = LN_model().load_from_checkpoint(model)
    else:
        model = LN_model()
    datamodule = Flowers102DataModule(predict_data=data)
    trainer = pl.Trainer()
    preds = trainer.predict(model=model, datamodule=datamodule)

    return preds[0].tolist()


if __name__ == "__main__":
    _predict()
