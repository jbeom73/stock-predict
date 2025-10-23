from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/add")
def add(a: int, b: int):
    return {"result": a + b}

@app.get("/api/oneplusone")
def oneplusone():
    return {"result": 1 + 1}
