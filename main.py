from fastapi import FastAPI
from pydantic import BaseModel
from algorithms import main as algorithms


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


class Coordinates(BaseModel):
    start: list
    end: list


"""
    {
    "coordinates": {
    "lat": 45.0,
    "lng": -123.0,}
    }
"""


@app.post("/map_route")
async def map_route(coordinates: Coordinates, algorithm: str = "busqueda_amplitud"):
    start = coordinates.start
    end = coordinates.end
    ruta = algorithms(algorithm, start, end)
    return {"algorithm": algorithm, "ruta": ruta}
