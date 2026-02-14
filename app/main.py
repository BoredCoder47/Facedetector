from fastapi import FastAPI

app = FastAPI(title="Anton")

@app.get("/")
async def root():
    return {"message": "Anton is monitoring you!"}