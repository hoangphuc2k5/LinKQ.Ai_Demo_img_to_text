# OCR Image to Text

Project chay PaddleOCR truc tiep trong Vercel Function. `app.py` duoc giu lai de chay va debug OCR o may local.

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

## Deploy full len Vercel

PaddleOCR duoc chay truc tiep trong `api/ocr.py`; khong can `OCR_BACKEND_URL` va khong goi sang OCR server ben ngoai. Hai model OCR da duoc dong goi trong thu muc `models/`, do do function khong tai model tu Internet khi cold start.

1. Import thu muc project nay vao Vercel.
2. Neu Vercel bao function vuot qua 250 MB, them Environment Variable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`, sau do redeploy.
3. Deploy. Web co URL `https://<ten-project>.vercel.app` va API la `https://<ten-project>.vercel.app/api/ocr`.

Khi develop, chay `vercel dev`. Vercel gioi han request tai function, nen giao dien chi nhan anh nho hon 4 MB. `app.py` co the duoc chay rieng bang `python app.py` de test nhanh o may local.
