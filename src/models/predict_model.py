import click
import pytorch_lightning as pl

from src.data.LN_data_module import Flowers102DataModule
from src.models.LN_model import LN_model

# Takes a model.ckpt and a folder with images and returns predicted classes.


@click.command()
@click.option(
    "--model",
    default="flowers/2w7od6l3/checkpoints/epoch=4-step=10.ckpt",
    help="Model checkpoint for prediction",
)
@click.option("--data", default="./predict", help="learning rate to use for training")
def predict(model, data):
    print("Predicting day and night")

    model = LN_model().load_from_checkpoint(model)
    datamodule = Flowers102DataModule(predict_dir=data)
    trainer = pl.Trainer(logger=pl.loggers.WandbLogger(project="flowers"))

    preds = trainer.predict(model=model, datamodule=datamodule)

    return f"Class predictions: {preds}"


if __name__ == "__main__":
    predict()
