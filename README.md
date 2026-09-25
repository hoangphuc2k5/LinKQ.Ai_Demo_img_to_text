# OCR Image to Text

Project co hai phan:

- `app.py`: OCR backend dung PaddleOCR (chay local, Docker, Render, Railway hoac VPS).
- `public/` va `api/`: frontend + API proxy de deploy len Vercel.

```powershell
python app.py
```

Mo `http://127.0.0.1:8000`, chon anh va bam **Doc anh**.

## API

`POST /api/ocr` voi body `multipart/form-data`, truong file la `image`.

```powershell
curl.exe -X POST -F "image=@duong-dan\anh.png" http://127.0.0.1:8000/api/ocr
```

API tra JSON gom `text`, `lines` (text, do tin cay, toa do), `count` va `processing_seconds`.

Anh duoc ho tro: PNG, JPG/JPEG, WebP, BMP, TIFF; gioi han 10 MB. Model OCR chi duoc khoi tao o request dau tien va duoc dung lai cho cac request sau.

## Deploy web len Vercel

PaddleOCR khong nen chay truc tiep tren Vercel; hay deploy `app.py` tai mot OCR backend co the chay Python/PaddleOCR. Sau do:

1. Import thu muc project nay vao Vercel.
2. Trong **Settings → Environment Variables**, them `OCR_BACKEND_URL`, vi du `https://ocr-api.example.com/api/ocr`.
3. Deploy. Web co URL `https://<ten-project>.vercel.app` va proxy API la `https://<ten-project>.vercel.app/api/ocr`.

Khi develop Vercel o may local, chay OCR backend trong mot terminal:

```powershell
python app.py
```

Tao file `.env.local` (khong commit) voi noi dung:

```text
OCR_BACKEND_URL=http://127.0.0.1:8000/api/ocr
```

Sau do chay `vercel dev`. Vercel gioi han request tai function, nen giao dien Vercel chi nhan anh nho hon 4 MB.
