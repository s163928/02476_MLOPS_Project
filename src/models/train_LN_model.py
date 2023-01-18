from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.data.LN_data_module import Flowers102DataModule
from src.models.LN_model import LN_model
from pytorch_lightning.callbacks import Callback
import wandb
import pytorch_lightning as pl
import hydra
# from hydra.utils import dir as dir_utils
import omegaconf
import pprint

class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)


            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            self.wandb_logger.log_table(key='sample_table', columns=columns, data=data)


@hydra.main(config_path="./../../configs/", config_name="defaults")
def main(cfg: omegaconf.DictConfig) -> None:

    # # Create a directory to save the data and model files
    # data_dir = dir_utils.get_path("data")
    # model_dir = dir_utils.get_path("models")

    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)

    print("Training day and night")
    pprint.pprint(cfg, indent=2)

    model = LN_model(model_name = cfg.model.model_name,
                    pretrained = cfg.model.pretrained,
                    in_chans = cfg.model.in_chans,
                    num_classes = cfg.model.num_classes,
                    task = cfg.training.task,
                    optimizer = cfg.training.optimizer, 
                    lr = cfg.optimizer.config.learning_rate, 
                    loss = cfg.training.loss,
                    logger = wandb_logger)
    data = Flowers102DataModule()
    early_stopping = EarlyStopping(cfg.training.callbacks.EarlyStopping.monitor)

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")
    
    # Callbacks
    log_predictions_callback = LogPredictionsCallback()

    # Overwrites checkpoint if improvement over epoch
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.callbacks.ModelCheckpoint.monitor,
        dirpath=cfg.training.model_dir_name,
        filename=cfg.training.model_output_name,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        limit_train_batches = cfg.training.limit_train_batches,  # Limit to 20% of total size.
        max_epochs = cfg.training.max_epochs,
        logger = wandb_logger,
        log_every_n_steps = cfg.training.logging.log_every_n_steps,
        # callbacks = [early_stopping, checkpoint_callback],
        callbacks=[early_stopping, checkpoint_callback, log_predictions_callback]
    )
    trainer.fit(model = model, datamodule = data)

if __name__ == "__main__":
    main()
