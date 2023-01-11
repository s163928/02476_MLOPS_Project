import pytorch_lightning as pl
from src.models.LN_model import LN_model
from src.data.LN_data_module import Flowers102DataModule
import torch
# Should take a model.pt and either a folder with images or already loaded image in numpy or pickle.

def main():
    print("Predicting day and night")

    model = LN_model().load_from_checkpoint("./flowers/1wczfaxo/checkpoints/epoch=4-step=10.ckpt")
    data = Flowers102DataModule()

    trainer = pl.Trainer()
    
    trainer.predict(
        model=model,
        datamodule=data)
    

if __name__ == "__main__":
    main()