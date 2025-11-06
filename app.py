from main import graph

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/summarizer")
def video_summarizer(video_url: str):
    result = graph.invoke(input={"video_url": video_url})
    return result