"""
FastAPI 앱 엔트리포인트.

실행:
  개발:  uvicorn web.main:app --reload
  운영:  uvicorn web.main:app --host 0.0.0.0 --port 8000
"""

"""
  GET  /api                                # 목록 + 로드 상태                 
  POST /api/{name}/process-image/          # {name} = vgg16 | resnet50
  POST /api/{name}/process-image/crop/                                        
  POST /compare/process-image/                   # ★ 병렬 실행 + latencyMs
  POST /visualize/                               # YOLO (백본 독립)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import STATIC_DIR
from web.routes import compare, features, visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(
    title="Image Feature Extraction Service",
    description="Multi-backbone (VGG16 / ResNet-50) 이미지 특징점 추출 + 비교",
)

# 라우터 등록
app.include_router(visualize.router)
app.include_router(features.router, prefix="/api")
app.include_router(compare.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
