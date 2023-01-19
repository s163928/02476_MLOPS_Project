from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from src.models.predict_model import predict as model_predict
from google.cloud import storage
from io import BytesIO

from opentelemetry import trace

from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# from opentelemetry.propagate import set_global_textmap
# from opentelemetry.propagators.cloud_trace_propagator import (
#    CloudTraceFormatPropagator,
# )

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
with tracer.start_as_current_span("foo"):
    print("Hello world!")

storage_client = storage.Client()
bucket = storage_client.get_bucket("mlops-project")
blob = bucket.blob("jobs/vertex-with-docker/model.ckpt")
checkpoint_data = blob.download_as_string()

app = FastAPI(title="FastAPI")
FastAPIInstrumentor.instrument_app(app)

# set_global_textmap(CloudTraceFormatPropagator())


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
    with tracer.start_as_current_span("img-read") as img_read_span:
        img_read_span.set_attribute("img-filename", data.filename)
        img_data = await data.read()

    with tracer.start_as_current_span("predict-result") as predit_span:
        preds = model_predict(data=img_data, model=BytesIO(checkpoint_data))
        predit_span.set_attribute("prediction", preds)
        response = {
            "input": data,
            "pred": preds,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }
    return response
