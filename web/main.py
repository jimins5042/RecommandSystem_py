import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from shared import STATIC_DIR
import vgg16
import efficientNet

app = FastAPI()

app.include_router(vgg16.router)
app.include_router(efficientNet.router)

# static 파일 서빙 (라우터 등록 후 마운트)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
