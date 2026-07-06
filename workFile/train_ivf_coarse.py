"""
IVF coarse quantizer 학습 + coarse_id backfill 스크립트.

PQ 코드북(train_pq_codebook.py)과 달리, coarse 셀은 "실제 색인 대상 데이터를
어떻게 분할하느냐"라서 학습용 이미지셋 캐시가 아니라 **DB(image_info)에 실제로
적재된 임베딩**으로 학습해야 recall 에 유리하다.

흐름:
    1. image_info 에서 (image_uuid, embedding_value) 전량 로드 → fp16 → float32 (N, 2048)
    2. FAISS k-means 로 nlist 개 coarse centroid 학습
    3. centroid → model/coarse.npy 저장   (Spring PQFiltering.loadCoarseCentroids 가 읽음)
    4. 각 벡터의 최근접 centroid = coarse_id → UPDATE image_info SET coarse_id

산출물:
    model/coarse.npy   — shape (nlist, 2048) float32, C-order, '<f4'

사전 준비 (Oracle):
    ALTER TABLE image_info ADD (coarse_id NUMBER(10));
    CREATE INDEX idx_image_info_coarse ON image_info (coarse_id);

.env (config.py 가 load_dotenv 로 읽음 — Spring 과 동일 키 재사용):
    db_username = ADMIN
    db_password = ****
    db_url = jdbc:oracle:thin:@l27rbzwhpce8ao9l_high?TNS_ADMIN=<wallet_dir>
    db_wallet_password = ****     # OCI wallet 다운로드 시 설정한 비번 (신규 추가 필요)

사용 예:
    pip install oracledb
    python workFile/train_ivf_coarse.py --nlist 256
    python workFile/train_ivf_coarse.py --nlist 512 --dry-run   # UPDATE 없이 학습만
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path 에 추가 (train_pq_codebook.py 와 동일)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODEL_DIR  # noqa: E402  (.env 로드 side-effect 포함)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 2048
COARSE_PATH = Path(MODEL_DIR) / "coarse.npy"

# fetch 시 메모리/왕복 균형. 수십만 건이면 이 단위로 나눠 읽음.
FETCH_BATCH = 10_000
UPDATE_BATCH = 5_000


def get_connection():
    """python-oracledb thin 모드로 OCI Autonomous DB(mTLS wallet) 연결.

    Spring 과 동일한 .env 키를 재사용:
        db_username / db_password
        db_url = jdbc:oracle:thin:@<alias>?TNS_ADMIN=<wallet_dir>
    추가로 필요 (암호화된 ewallet.pem 복호화용):
        db_wallet_password = OCI wallet 다운로드 시 설정한 비밀번호
    """
    try:
        import oracledb
    except ImportError as e:
        raise ImportError("oracledb 미설치 — `pip install oracledb`") from e

    user = os.getenv("db_username")
    password = os.getenv("db_password")
    jdbc_url = os.getenv("db_url")
    wallet_pw = os.getenv("db_wallet_password")
    if not (user and password and jdbc_url):
        raise RuntimeError("db_username / db_password / db_url 환경변수(.env) 필요")

    # jdbc:oracle:thin:@<alias>?TNS_ADMIN=<dir> 에서 alias 와 wallet 디렉토리 추출
    m = re.search(r"@([^?]+)\?TNS_ADMIN=(.+)$", jdbc_url.strip())
    if not m:
        raise RuntimeError(f"db_url 파싱 실패(@alias?TNS_ADMIN=dir 형식 필요): {jdbc_url}")
    alias, wallet_dir = m.group(1).strip(), m.group(2).strip()
    if not wallet_pw:
        raise RuntimeError(
            "db_wallet_password 필요 — ewallet.pem 이 암호화돼 있어 thin 모드는 "
            "wallet 비밀번호가 있어야 함(.env 에 추가)."
        )

    # BLOB 을 LOB 객체 대신 bytes 로 바로 받기
    oracledb.defaults.fetch_lobs = False
    logger.info(f"Oracle 연결(thin/wallet): {user}@{alias}  wallet={wallet_dir}")
    return oracledb.connect(
        user=user,
        password=password,
        dsn=alias,                    # tnsnames.ora 의 별칭
        config_dir=wallet_dir,        # tnsnames.ora / sqlnet.ora 위치
        wallet_location=wallet_dir,   # ewallet.pem 위치 (thin 모드)
        wallet_password=wallet_pw,    # 암호화된 PEM 복호화
    )


def load_embeddings(conn) -> tuple[list[str], np.ndarray]:
    """image_info 에서 pq_code 가 적재된 행의 (uuid, fp16 임베딩) 로드.

    PQ 검색 대상(pq_code IS NOT NULL)과 동일 집합만 색인 → 셀 분포 일관.
    """
    cur = conn.cursor()
    cur.arraysize = FETCH_BATCH
    cur.execute(
        "SELECT image_uuid, embedding_value "
        "FROM image_info "
        "WHERE embedding_value IS NOT NULL AND pq_code IS NOT NULL"
    )

    uuids: list[str] = []
    chunks: list[np.ndarray] = []
    start = time.time()
    while True:
        rows = cur.fetchmany(FETCH_BATCH)
        if not rows:
            break
        for uuid, blob in rows:
            if blob is None or len(blob) != EMBEDDING_DIM * 2:  # fp16 = 2 byte
                logger.warning(f"  skip {uuid}: blob len={None if blob is None else len(blob)}")
                continue
            uuids.append(uuid)
            chunks.append(np.frombuffer(blob, dtype=np.float16).astype(np.float32))
        logger.info(f"  로드 {len(uuids)}건...")
    cur.close()

    if not uuids:
        raise RuntimeError("임베딩이 하나도 로드되지 않음 (image_info 확인)")

    X = np.ascontiguousarray(np.stack(chunks), dtype=np.float32)
    logger.info(f"임베딩 로드 완료: {X.shape} ({time.time() - start:.1f}s)")
    return uuids, X


def train_coarse(X: np.ndarray, nlist: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """FAISS k-means 로 coarse centroid 학습.

    반환:
        centroids  (nlist, D) float32
        labels     (N,)       int32  — 각 벡터의 최근접 centroid = coarse_id
    학습/할당 모두 L2 (Java selectProbes 의 L2² probe 와 일치 — raw 벡터 사용, 정규화 X).
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss 미설치 — requirements.txt 의 faiss-cpu 확인") from e

    n = len(X)
    # k-means 안정성: centroid 당 최소 수십 포인트 권장. 과하면 자동 경고.
    if n < nlist * 39:
        logger.warning(
            f"학습 샘플 {n} < nlist*39 ({nlist * 39}) — nlist 를 줄이는 게 나을 수 있음."
        )

    logger.info(f"k-means 학습: N={n}, D={EMBEDDING_DIM}, nlist={nlist}")
    km = faiss.Kmeans(EMBEDDING_DIM, nlist, niter=25, seed=seed, verbose=True)
    start = time.time()
    km.train(X)
    logger.info(f"k-means 완료 ({time.time() - start:.1f}s)")

    centroids = km.centroids.reshape(nlist, EMBEDDING_DIM).astype(np.float32)
    _, labels = km.index.search(X, 1)          # 최근접 1개
    labels = labels.reshape(-1).astype(np.int32)

    # 셀 분포 요약 (편중 확인)
    counts = np.bincount(labels, minlength=nlist)
    logger.info(
        f"셀 분포 — 비어있음 {int((counts == 0).sum())}/{nlist}, "
        f"min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}"
    )
    return centroids, labels


def backfill_coarse_ids(conn, uuids: list[str], labels: np.ndarray) -> None:
    """UPDATE image_info SET coarse_id = :cid WHERE image_uuid = :uuid (배치)."""
    cur = conn.cursor()
    sql = "UPDATE image_info SET coarse_id = :1 WHERE image_uuid = :2"
    start = time.time()
    total = len(uuids)
    for i in range(0, total, UPDATE_BATCH):
        batch = [(int(labels[j]), uuids[j]) for j in range(i, min(i + UPDATE_BATCH, total))]
        cur.executemany(sql, batch)
        conn.commit()
        logger.info(f"  UPDATE {min(i + UPDATE_BATCH, total)}/{total}")
    cur.close()
    logger.info(f"coarse_id backfill 완료 ({time.time() - start:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IVF coarse quantizer 학습 + backfill")
    p.add_argument("--nlist", type=int, default=256, help="coarse 셀 개수 (기본 256, ≈√N 권장)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="coarse.npy 저장만, DB UPDATE 스킵")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_connection()
    try:
        uuids, X = load_embeddings(conn)
        centroids, labels = train_coarse(X, args.nlist, args.seed)

        # ── centroid 저장 (Java loadCoarseCentroids 호환: (nlist, D) '<f4' C-order) ──
        COARSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(COARSE_PATH, np.ascontiguousarray(centroids, dtype=np.float32))
        logger.info(f"coarse centroid 저장: {COARSE_PATH} (shape {centroids.shape})")

        if args.dry_run:
            logger.info("--dry-run: DB UPDATE 스킵")
        else:
            backfill_coarse_ids(conn, uuids, labels)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
