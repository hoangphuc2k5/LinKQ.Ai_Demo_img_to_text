import os
import sys


def main():
    print("=" * 60)
    print("KIỂM TRA CÀI ĐẶT PADDLEOCR")
    print("=" * 60)

    try:
        import paddleocr
        print(f"[OK] PaddleOCR version: {paddleocr.__version__}")
    except Exception as e:
        print(f"[LỖI] Không import được PaddleOCR: {e}")
        sys.exit(1)

    try:
        import paddle
        print(f"[OK] PaddlePaddle version: {paddle.__version__}")
        print(f"[OK] Paddle compiled with CUDA: {paddle.is_compiled_with_cuda()}")
        if paddle.is_compiled_with_cuda():
            print(f"[OK] GPU count: {paddle.device.cuda.device_count()}")
    except Exception as e:
        print(f"[LỖI] Không import được PaddlePaddle: {e}")
        sys.exit(1)

    try:
        import paddlex
        print(f"[OK] PaddleX version: {paddlex.__version__ if hasattr(paddlex, '__version__') else 'installed'}")
    except Exception as e:
        print(f"[LỖI] Không import được PaddleX: {e}")
        sys.exit(1)

    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"[CẢNH BÁO] Không import được OpenCV: {e}")

    try:
        from PIL import Image
        print("[OK] Pillow (PIL) installed")
    except Exception as e:
        print(f"[CẢNH BÁO] Không import được Pillow: {e}")

    try:
        import numpy
        print(f"[OK] NumPy version: {numpy.__version__}")
    except Exception as e:
        print(f"[LỖI] Không import được NumPy: {e}")
        sys.exit(1)

    print("-" * 60)
    vi_dict_path = os.path.join(
        os.path.dirname(__file__),
        "PaddleOCR",
        "ppocr",
        "utils",
        "dict",
        "vi_dict.txt",
    )
    if os.path.exists(vi_dict_path):
        print(f"[OK] Từ điển Tiếng Việt tìm thấy: {vi_dict_path}")
        with open(vi_dict_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[OK] Số lượng ký tự trong từ điển tiếng Việt: {len(lines)}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy vi_dict.txt tại: {vi_dict_path}")

    print("-" * 60)
    print("TẤT CẢ KIỂM TRA ĐÃ HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
