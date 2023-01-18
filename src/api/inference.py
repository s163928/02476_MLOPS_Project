from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from src.models.predict_model import predict as model_predict
from google.cloud import storage
from io import BytesIO

storage_client = storage.Client()
bucket = storage_client.get_bucket("mlops-project")
blob = bucket.blob("jobs/vertex-with-docker/model.ckpt")
checkpoint_data = blob.download_as_string()

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
async def predict(data: UploadFile = File(...)):
    img_data = await data.read()
    preds = model_predict(data=img_data, model=BytesIO(checkpoint_data))
    response = {
        "input": data,
        "pred": preds,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
