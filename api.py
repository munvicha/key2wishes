import re
import torch
import uvicorn
import pipeline
import time
from pipeline import generate
from fastapi import FastAPI, Query
from typing import List
import pydantic
from pydantic import BaseModel

app = FastAPI()

def gen(kws):
    return generate(keywords=kws)

@app.get("/")
def read_root():
    return {"Hello": "World"}


#@app.post("/api")
#def k2w_post(data: List[str]):
#    start_time = time.time()
#    return {
#        "keywords": data,
#        "text": generate(data),
#        "time": round((time.time() - start_time), 2)
#    }


class Item(BaseModel):
    kws: List[str]

@app.post("/api")
def k2w_get(data: Item):
    start_time = time.time()
    d = data.kws
    return {
        "keywords": d,
        "text": generate(d),
        "time": round((time.time() - start_time), 2)
    }