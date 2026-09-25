"""Vercel Function OCR chay PaddleOCR truc tiep, khong goi OCR backend ben ngoai."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/tmp/.paddlex")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_new_executor", "0")

MAX_BODY_BYTES = 4 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_ocr: Any | None = None
_ocr_lock = threading.Lock()
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


def get_ocr() -> Any:
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from paddleocr import PaddleOCR

            det_model = MODEL_ROOT / "PP-OCRv6_medium_det"
            rec_model = MODEL_ROOT / "PP-OCRv6_medium_rec"
            if not det_model.is_dir() or not rec_model.is_dir():
                raise RuntimeError(
                    f"OCR models are missing: det={det_model.exists()}, rec={rec_model.exists()}"
                )
            _ocr = PaddleOCR(
                text_detection_model_dir=str(det_model),
                text_recognition_model_dir=str(rec_model),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=2,
            )
    return _ocr


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    if "boundary=" not in content_type:
        raise ValueError("Content-Type phai la multipart/form-data.")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    delimiter = b"--" + boundary.encode("utf-8")
    for part in body.split(delimiter):
        if b'name="image"' not in part or b"\r\n\r\n" not in part:
            continue
        header, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]
        marker = b'filename="'
        if marker not in header:
            raise ValueError("Truong image khong co ten file.")
        filename = header.split(marker, 1)[1].split(b'"', 1)[0].decode("utf-8", "replace")
        if filename and content:
            return filename, content
    raise ValueError("Khong tim thay truong image hop le.")


def recognize(image_path: str) -> dict[str, Any]:
    started = time.perf_counter()
    lines: list[dict[str, Any]] = []
    for page in get_ocr().predict(image_path):
        data = getattr(page, "json", {}).get("res", {})
        for text, score, polygon in zip(
            data.get("rec_texts", []), data.get("rec_scores", []), data.get("rec_polys", [])
        ):
            text = str(text).strip()
            if text:
                lines.append({"text": text, "confidence": round(float(score), 4), "polygon": polygon})
    return {
        "text": "\n".join(item["text"] for item in lines),
        "lines": lines,
        "count": len(lines),
        "processing_seconds": round(time.perf_counter() - started, 2),
    }


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.send_json({"ok": True, "service": "PaddleOCR on Vercel"})

    def do_POST(self) -> None:
        temp_path = ""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("Anh phai nho hon 4 MB khi gui qua Vercel.")
            filename, content = parse_multipart(self.rfile.read(length), self.headers.get("Content-Type", ""))
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise ValueError("Dinh dang anh chua duoc ho tro.")
            with tempfile.NamedTemporaryFile(dir="/tmp", suffix=suffix, delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            response = recognize(temp_path)
            response["filename"] = Path(filename).name
            self.send_json(response)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            print(f"OCR error: {error}", flush=True)
            traceback.print_exc()
            self.send_json({"error": "Khong the doc anh. Xem Function Logs tren Vercel."}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
