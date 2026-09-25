"""Vercel Function: chuyen request upload toi OCR backend.

Dat OCR_BACKEND_URL trong Vercel, vi du:
https://ocr.example.com/api/ocr
"""

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_BODY_BYTES = 4 * 1024 * 1024  # Nho hon gioi han payload 4.5 MB cua Vercel.


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        backend_url = os.environ.get("OCR_BACKEND_URL")
        if not backend_url:
            self.send_json(
                {"error": "OCR_BACKEND_URL chua duoc cau hinh tren Vercel."},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= MAX_BODY_BYTES:
                raise ValueError("Anh phai nho hon 4 MB khi gui qua Vercel.")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Content-Type phai la multipart/form-data.")

            request = Request(
                backend_url,
                data=self.rfile.read(content_length),
                headers={"Content-Type": content_type, "Accept": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=55) as response:
                payload = response.read()
                status = response.status
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except HTTPError as error:
            self.send_json(
                {"error": f"OCR backend tra ve loi {error.code}."},
                HTTPStatus.BAD_GATEWAY,
            )
        except URLError:
            self.send_json(
                {"error": "Khong ket noi duoc OCR backend."},
                HTTPStatus.BAD_GATEWAY,
            )
        except Exception:
            self.send_json({"error": "Khong the xu ly anh."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        self.send_json({"ok": True, "service": "Vercel OCR proxy"})
