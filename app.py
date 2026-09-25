"""Trang web va API OCR don gian, su dung PaddleOCR da cai trong project.

Chay: python app.py
Mo:  http://127.0.0.1:8000
API:  POST /api/ocr (multipart/form-data, truong `image`)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# Tranh loi oneDNN/PIR tung duoc xu ly trong script OCR mau.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_new_executor", "0")

HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_ocr: Any | None = None
_ocr_lock = threading.Lock()


def get_ocr() -> Any:
    """Khoi tao model mot lan, khong tai lai cho moi request."""
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from paddleocr import PaddleOCR

            _ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                engine="paddle_static",
                enable_mkldnn=False,
                cpu_threads=4,
            )
    return _ocr


def extract_ocr(image_path: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = get_ocr().predict(image_path)
    lines: list[dict[str, Any]] = []

    for page in result:
        # PaddleOCR 3.8 tra OCRResult co .json["res"]. Cac ban khac co
        # the tra .json_items, nen ho tro ca hai de API de nang cap.
        json_items = getattr(page, "json_items", None)
        if json_items is not None:
            records = json_items
        else:
            page_result = getattr(page, "json", {}).get("res", {})
            records = [
                {"rec_text": text, "rec_score": score, "det_poly": polygon}
                for text, score, polygon in zip(
                    page_result.get("rec_texts", []),
                    page_result.get("rec_scores", []),
                    page_result.get("rec_polys", []),
                )
            ]
        for item in records:
            text = str(item.get("rec_text", "")).strip()
            if text:
                lines.append({
                    "text": text,
                    "confidence": round(float(item.get("rec_score", 0)), 4),
                    "polygon": item.get("det_poly", []),
                })

    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "count": len(lines),
        "processing_seconds": round(time.perf_counter() - started, 2),
    }


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Doc field `image` trong multipart/form-data, khong can framework ngoai."""
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Content-Type phai la multipart/form-data.")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    delimiter = b"--" + boundary.encode("utf-8")

    for part in body.split(delimiter):
        if b'name="image"' not in part or b"\r\n\r\n" not in part:
            continue
        header, content = part.split(b"\r\n\r\n", 1)
        # Sau moi part cua multipart co CRLF; chi bo hai byte phan cach,
        # khong dung rstrip() de tranh lam mat byte hop le cua anh.
        if content.endswith(b"\r\n"):
            content = content[:-2]
        filename_marker = b'filename="'
        if filename_marker not in header:
            raise ValueError("Truong image khong co ten file.")
        filename = header.split(filename_marker, 1)[1].split(b'"', 1)[0].decode("utf-8", "replace")
        if not filename or not content:
            raise ValueError("Vui long chon mot anh hop le.")
        return filename, content
    raise ValueError("Khong tim thay truong image trong form.")


class OCRHandler(BaseHTTPRequestHandler):
    server_version = "SimpleOCR/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/api/ocr":
            self.send_json({"error": "Khong tim thay endpoint."}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= MAX_UPLOAD_BYTES:
                raise ValueError("Anh phai nho hon 10 MB.")
            filename, content = parse_multipart(self.rfile.read(content_length), self.headers.get("Content-Type", ""))
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise ValueError("Dinh dang anh chua duoc ho tro.")

            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                response = extract_ocr(temp_path)
                response["filename"] = Path(filename).name
                self.send_json(response)
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            print(f"OCR error: {error}")
            self.send_json({"error": "Khong the doc anh. Vui long thu lai."}, HTTPStatus.INTERNAL_SERVER_ERROR)


INDEX_HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR - Đọc chữ từ ảnh</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#172033;font:16px system-ui,sans-serif}.wrap{max-width:820px;margin:52px auto;padding:0 20px}h1{margin:0 0 8px}p{color:#62708a}.card{margin-top:28px;padding:25px;background:#fff;border-radius:16px;box-shadow:0 8px 30px #1d2b4d16}input{display:block;width:100%;padding:14px;border:1px dashed #96a5c3;border-radius:10px}button{margin-top:15px;padding:12px 22px;border:0;border-radius:9px;background:#2359d9;color:#fff;font-weight:650;cursor:pointer}button:disabled{background:#94a3bd}#preview{max-width:100%;max-height:300px;display:none;margin-top:20px;border-radius:10px}#status{margin-top:16px;color:#52617c}pre{white-space:pre-wrap;min-height:100px;padding:16px;background:#f2f5fb;border-radius:10px;font:15px ui-monospace,monospace}.meta{font-size:14px;color:#62708a}
</style></head><body><main class="wrap"><h1>Đọc chữ từ ảnh</h1><p>Upload PNG, JPG, WebP, BMP hoặc TIFF (tối đa 10 MB).</p><section class="card"><input id="image" type="file" accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"><img id="preview" alt="Ảnh xem trước"><button id="submit">Đọc ảnh</button><p id="status"></p><div id="result" hidden><p class="meta" id="meta"></p><pre id="text"></pre></div></section></main><script>
const input=document.querySelector('#image'), preview=document.querySelector('#preview'), button=document.querySelector('#submit'), status=document.querySelector('#status'), result=document.querySelector('#result');
input.onchange=()=>{const f=input.files[0];preview.style.display=f?'block':'none';if(f)preview.src=URL.createObjectURL(f)};
button.onclick=async()=>{const f=input.files[0];if(!f){status.textContent='Hãy chọn ảnh trước.';return}const form=new FormData();form.append('image',f);button.disabled=true;result.hidden=true;status.textContent='Đang nhận dạng… Lần đầu có thể mất thời gian tải model.';try{const r=await fetch('/api/ocr',{method:'POST',body:form}), data=await r.json();if(!r.ok)throw Error(data.error||'Có lỗi xảy ra');document.querySelector('#text').textContent=data.text||'(Không nhận diện được văn bản)';document.querySelector('#meta').textContent=`${data.count} dòng · ${data.processing_seconds}s`;result.hidden=false;status.textContent='Hoàn tất.'}catch(e){status.textContent=e.message}finally{button.disabled=false}};
</script></body></html>'''


if __name__ == "__main__":
    print(f"OCR web dang chay tai http://{HOST}:{PORT}")
    print("Nhan Ctrl+C de dung server.")
    ThreadingHTTPServer((HOST, PORT), OCRHandler).serve_forever()
