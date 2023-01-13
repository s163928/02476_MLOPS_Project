from fastapi import FastAPI
from http import HTTPStatus
from fastapi import UploadFile, File
from src.models.predict_model import predict as model_predict

app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/prediction/")
async def predict(data: UploadFile = File(...), model: str = "models/model.ckpt"):
    img_data = await data.read()
    preds = model_predict(data=img_data, model=model)
    response = {
        "input": data,
        "pred": preds,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
