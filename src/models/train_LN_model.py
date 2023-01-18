import pytorch_lightning as pl
from src.models.LN_model import LN_model
from src.data.LN_data_module import Flowers102DataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import shutil
import os

def main():
    print("Training day and night")

    model = LN_model()
    data = Flowers102DataModule()
    early_stopping = EarlyStopping("val_loss")

    # Overwrites checkpoint if improvement over epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models',
        filename='model',
        auto_insert_metric_name=False
        )

    trainer = pl.Trainer(
        limit_train_batches=0.20, # Limit to 20% of total size.
        max_epochs=5,
        logger=pl.loggers.WandbLogger(project="flowers"),
        log_every_n_steps=1,
        callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(
        model=model,
        datamodule=data)

    upload_model()

def upload_model(model_name = 'model.ckpt',
    model_path = "./models",
    bucket_name = "/gcs/mlops-project/jobs/training/vertex-with-docker",
    mode = "vertex-job"):

    if mode!= "vertex-job":
        from google.cloud import storage

        # Create a new client
        storage_client = storage.Client()

        # Set the name of the new bucket
        bucket_name = bucket_name

        try:
            # Create the new bucket
            bucket = storage_client.create_bucket(bucket_name)
            print("Bucket {} created.".format(bucket.name))
        except Exception as e:
            print(e)
            bucket = storage_client.get_bucket(bucket_name)
            
        # Upload a file to the new bucket
        blob = bucket.blob(model_name)
        blob.upload_from_filename(os.path.join(model_path, model_name))
        print("File uploaded to {}.".format(blob.name))
    
    else:
        # copy model file after training to gcp-bucket
        gcp_bucket = '/gcs/mlops-project/jobs/vertex-with-docker'
        model_dir = '/models'
        shutil.copy2(os.path.join(model_dir,model_name), gcp_bucket)
        print("Model saved to GCP Bucket")

if __name__ == "__main__":
    main()



