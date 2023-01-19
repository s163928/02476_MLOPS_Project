from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from src.models.predict_model import predict as model_predict
from google.cloud import storage
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)


from opentelemetry import trace

from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# from opentelemetry.propagate import set_global_textmap
# from opentelemetry.propagators.cloud_trace_propagator import (
#    CloudTraceFormatPropagator,
# )

NUM_FEATURES = 100
PREDICTION_BUCKET = "prediction_database"
DATABASE_FILE = "prediction_database.csv"

# set up tracing and open telemetry
provider = TracerProvider()
trace.set_tracer_provider(provider)

cloud_trace_exporter = CloudTraceSpanExporter(
    project_id="primal-graph-374308",
)
provider.add_span_processor(
    # BatchSpanProcessor buffers spans and sends them in batches in a
    # background thread. The default parameters are sensible, but can be
    # tweaked to optimize your performance
    BatchSpanProcessor(cloud_trace_exporter)
)

tracer = trace.get_tracer(__name__)

storage_client = storage.Client()

bucket = storage_client.get_bucket("mlops-project")
blob = bucket.blob("jobs/vertex-with-docker/model.ckpt")
checkpoint_data = blob.download_as_bytes()

app = FastAPI(title="FastAPI")
FastAPIInstrumentor.instrument_app(app)

# set_global_textmap(CloudTraceFormatPropagator())


def add_to_database(
    img_data,
    pred,
    storage_client,
    bucket_name: str,
    database_file: str,
    num_features: int,
):
    img = Image.open(BytesIO(img_data))
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(database_file)
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
    )

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs["pixel_values"])

    feature_list = [str(x) for x in img_features.view(-1).tolist()[:NUM_FEATURES]]

    if not blob.exists():
        with open(DATABASE_FILE, "w") as file:
            print(
                "target",
                ",".join(["feature_" + str(i) for i in range(NUM_FEATURES)]),
                sep=",",
                file=file,
            )
    else:
        blob.download_to_filename(DATABASE_FILE)

    with open(DATABASE_FILE, "a") as file:
        print(pred, ",".join(feature_list), sep=",", file=file)
    blob.upload_from_filename(DATABASE_FILE)


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/prediction/")
async def predict(background_tasks: BackgroundTasks, data: UploadFile = File(...)):
    with tracer.start_as_current_span("img-read") as img_read_span:
        img_read_span.set_attribute("img-filename", data.filename)
        img_data = await data.read()

    with tracer.start_as_current_span("predict-result") as predit_span:
        pred = model_predict(data=img_data, model=BytesIO(checkpoint_data))[0]
        predit_span.set_attribute("prediction", pred)
        response = {
            "input": data,
            "pred": pred,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }

    background_tasks.add_task(
        add_to_database,
        img_data,
        pred,
        storage_client,
        PREDICTION_BUCKET,
        DATABASE_FILE,
        NUM_FEATURES,
    )

    return response


@app.get("/monitoring/", response_class=HTMLResponse)
async def monitoring():
    bucket = storage_client.get_bucket("prediction_database")
    prediction_data = bucket.blob("prediction_database.csv").download_as_bytes()
    reference_data = bucket.blob("reference_database.csv").download_as_bytes()

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(
        current_data=pd.read_csv(BytesIO(prediction_data)),
        reference_data=pd.read_csv(BytesIO(reference_data)),
        column_mapping=None,
    )
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
