"""Probe the local BGE-M3 ONNX export on CPU."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

model_path = Path(r"D:\hermes\models\bge_m3_onnx_export\model.onnx")
tokenizer_path = Path(r"D:\hermes\models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181")
started = time.perf_counter()
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
loaded = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
encoded = tokenizer("Codex Obsidian memory CPU semantic retrieval", return_tensors="np", truncation=True, max_length=128)
inputs = {key: value.astype(np.int64) for key, value in encoded.items() if key in {item.name for item in session.get_inputs()}}
outputs = session.run(None, inputs)
vector = np.asarray(outputs[0])
print(json.dumps({"provider": "onnxruntime", "model": "BAAI/bge-m3", "load_seconds": round(loaded-started, 2), "encode_seconds": round(time.perf_counter()-loaded, 2), "shape": list(vector.shape), "status": "PASS"}, ensure_ascii=False))
