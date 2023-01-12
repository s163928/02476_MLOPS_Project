from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re
from pydantic import BaseModel
from fastapi import UploadFile, File
from typing import Optional
from fastapi.responses import FileResponse


app = FastAPI()

@app.get("/")
def root():
	""" Health check."""
	response = {
		"message": HTTPStatus.OK.phrase,
		"status-code": HTTPStatus.OK,
	}
	return response

