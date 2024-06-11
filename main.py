import os
import json
import uvicorn
import gdown

from typing import List
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from utils import get_prediction


app = FastAPI()

with open("config.json") as json_file:
    config = json.load(json_file)

if not os.path.isdir(config['model_path']):
    gdown.download_folder(config['drive_path'], quiet=True)

tokenizer = AutoTokenizer.from_pretrained(config['model_path'], model_max_length=512)
model = AutoModelForTokenClassification.from_pretrained(config['model_path'])
pipe = pipeline(model=model, tokenizer=tokenizer, 
    task='ner', aggregation_strategy='average')


class NerRequest(BaseModel):
    text: str


class NerResponse(BaseModel):
    prediction: List[str]


@app.post("/predict", response_model=NerResponse)
def predict(request: NerRequest):
    prediction = get_prediction(request.text, pipe(request.text))

    return NerResponse(
        prediction=prediction
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", log_level="info")