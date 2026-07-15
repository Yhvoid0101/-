"""Probe the locally cached BGE-M3 CPU provider without mutating the runtime."""
import sys
import time
import traceback
from pathlib import Path

from sentence_transformers import SentenceTransformer

model_path = Path(r"D:\hermes\models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181")
started = time.perf_counter()
try:
    model = SentenceTransformer(str(model_path), device="cpu")
    loaded = time.perf_counter()
    vector = model.encode(["Codex Obsidian memory CPU semantic retrieval"], normalize_embeddings=True, show_progress_bar=False)
    print({"provider": "sentence-transformers", "model": "BAAI/bge-m3", "load_seconds": round(loaded - started, 2), "encode_seconds": round(time.perf_counter() - loaded, 2), "dimension": len(vector[0]), "status": "PASS"})
except BaseException:
    traceback.print_exc()
    sys.exit(2)
