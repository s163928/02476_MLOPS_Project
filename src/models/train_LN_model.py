import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.LN_data_module import Flowers102DataModule
from src.models.LN_model import LN_model


def main():
    print("Training day and night")

    model = LN_model()
    data = Flowers102DataModule()
    early_stopping = EarlyStopping("val_loss")

    # Overwrites checkpoint if improvement over epoch
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",
        filename="model",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        limit_train_batches=0.20,  # Limit to 20% of total size.
        max_epochs=5,
        logger=pl.loggers.WandbLogger(project="flowers"),
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
