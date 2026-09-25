import os
import sys
import json
import time

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_new_executor", "0")


def _set_env_for_fix():
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["FLAGS_enable_new_executor"] = "0"


def _print_env_flags():
    flags = ["FLAGS_use_mkldnn", "FLAGS_enable_pir_api", "FLAGS_enable_new_executor"]
    for f in flags:
        print(f"  -> {f} = {os.environ.get(f, '<unset>')}")


def run_ocr(image_path, output_dir="output", use_latin_model=False, engine="paddle_static", cpu_threads=None):
    _set_env_for_fix()

    from paddleocr import PaddleOCR

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PADDLEOCR - NHẬN DẠNG CHỮ VIẾT (OCR) TIẾNG VIỆT + TIẾNG ANH")
    print("=" * 70)
    print(f"\nCấu hình môi trường (fix lỗi oneDNN / PIR):")
    _print_env_flags()

    print(f"\n[1/3] Khởi tạo PaddleOCR pipeline (engine={engine})...")
    start_init = time.time()

    common_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "engine": engine,
        "enable_mkldnn": False,
    }
    if cpu_threads and cpu_threads > 0:
        common_kwargs["cpu_threads"] = int(cpu_threads)

    if use_latin_model:
        print("  -> Model: PP-OCRv5 Latin (mobile - tối ưu cho tiếng Việt/Anh)")
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            **common_kwargs,
        )
        model_desc = "PP-OCRv5_mobile_det + latin_PP-OCRv5_mobile_rec"
    else:
        print("  -> Model: PP-OCRv6 Medium (mặc định - 50 ngôn ngữ)")
        ocr = PaddleOCR(**common_kwargs)
        model_desc = "PP-OCRv6_medium (50 languages including Vietnamese)"

    init_time = time.time() - start_init
    print(f"  -> Khởi tạo xong trong {init_time:.2f}s")

    if not os.path.exists(image_path):
        print(f"\n[LỖI] Không tìm thấy ảnh: {image_path}")
        sys.exit(1)

    print(f"\n[2/3] Thực hiện OCR với ảnh: {image_path}")
    start_predict = time.time()

    try:
        result = ocr.predict(image_path)
    except Exception as e:
        print(f"[LỖI] OCR thất bại (engine={engine}): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    predict_time = time.time() - start_predict
    print(f"  -> OCR xong trong {predict_time:.2f}s")

    print(f"\n[3/3] Kết quả OCR:")
    print("-" * 70)

    all_records = []
    total_texts = 0

    for idx, res in enumerate(result):
        print(f"\nẢnh #{idx + 1}: {getattr(res, 'img_name', os.path.basename(image_path))}")

        json_items = getattr(res, "json_items", None)
        if json_items:
            texts = []
            for item in json_items:
                rec_text = item.get("rec_text", "")
                rec_score = item.get("rec_score", 0.0)
                det_poly = item.get("det_poly", [])

                if rec_text:
                    total_texts += 1
                    texts.append(rec_text)
                    bbox_str = ""
                    if det_poly:
                        try:
                            xs = [p[0] for p in det_poly]
                            ys = [p[1] for p in det_poly]
                            bbox_str = f" [bbox: x={min(xs):.0f},y={min(ys):.0f},w={max(xs)-min(xs):.0f},h={max(ys)-min(ys):.0f}]"
                        except Exception:
                            pass
                    print(f"  [{total_texts:3d}] ({rec_score:.4f}) {rec_text}{bbox_str}")

                    all_records.append({
                        "index": total_texts,
                        "text": rec_text,
                        "confidence": float(rec_score),
                        "polygon": det_poly,
                    })

            print("-" * 70)
            print("DỮ LIỆU ĐÃ TRÍCH XUẤT (text-only):")
            print("-" * 70)
            for t in texts:
                print(t)

        try:
            res.save_to_img(output_dir)
            print(f"\n[OK] Ảnh kết quả (với bboxes) đã lưu vào: {output_dir}/")
        except Exception as e:
            print(f"\n[CẢNH BÁO] Không lưu được ảnh kết quả: {e}")

        try:
            res.save_to_json(output_dir)
            print(f"[OK] JSON chi tiết đã lưu vào: {output_dir}/")
        except Exception as e:
            print(f"[CẢNH BÁO] Không lưu được JSON: {e}")

    print("-" * 70)
    print(f"TỔNG: {total_texts} dòng văn bản được nhận dạng")
    print(f"Thời gian: Init={init_time:.2f}s, Predict={predict_time:.2f}s, Tổng={init_time+predict_time:.2f}s")

    custom_json_path = os.path.join(output_dir, "ocr_result_vietnamese.json")
    with open(custom_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": image_path,
            "total_texts": total_texts,
            "init_time_seconds": round(init_time, 2),
            "predict_time_seconds": round(predict_time, 2),
            "engine": engine,
            "enable_mkldnn": False,
            "model_config": model_desc,
            "results": all_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Kết quả JSON tùy chỉnh: {custom_json_path}")

    return all_records


def list_sample_images():
    repo_dir = os.path.join(os.path.dirname(__file__), "PaddleOCR")
    candidates = [
        os.path.join(repo_dir, "docs", "images", "en_1.png"),
        os.path.join(repo_dir, "docs", "images", "en_2.png"),
        os.path.join(repo_dir, "docs", "images", "en_3.png"),
        os.path.join(repo_dir, "tests", "test_files", "book.jpg"),
        os.path.join(repo_dir, "tests", "test_files", "table.jpg"),
        os.path.join(repo_dir, "docs", "images", "PP-OCRv3-pic001.jpg"),
        os.path.join(repo_dir, "docs", "images", "PP-OCRv3-pic002.jpg"),
    ]
    print("Các ảnh mẫu có sẵn trong PaddleOCR repo:")
    found = []
    for p in candidates:
        if os.path.exists(p):
            found.append(p)
            size_kb = os.path.getsize(p) / 1024
            print(f"  [{len(found)}] {os.path.relpath(p, repo_dir)}  ({size_kb:.1f} KB)")
    return found


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaddleOCR - OCR Tiếng Việt + Tiếng Anh")
    parser.add_argument("-i", "--image", type=str, help="Đường dẫn ảnh cần OCR")
    parser.add_argument("-o", "--output", type=str, default="output", help="Thư mục lưu kết quả")
    parser.add_argument("--latin", action="store_true", help="Dùng PP-OCRv5 Latin model thay vì PP-OCRv6")
    parser.add_argument("--list", action="store_true", help="Liệt kê các ảnh mẫu có sẵn")
    parser.add_argument(
        "--engine",
        type=str,
        default="paddle_static",
        choices=["paddle", "paddle_static", "paddle_dynamic", "onnxruntime", "transformers"],
        help="Inference engine (default: paddle_static; nếu lỗi thử paddle_dynamic hoặc onnxruntime)",
    )
    parser.add_argument("--cpu-threads", type=int, default=4, help="Số luồng CPU cho inference (default: 4)")
    args = parser.parse_args()

    if args.list:
        list_sample_images()
        sys.exit(0)

    img_path = args.image
    if not img_path:
        samples = list_sample_images()
        if samples:
            img_path = samples[0]
            print(f"\nKhông có -i/--image, sử dụng ảnh mẫu mặc định: {img_path}")
        else:
            print("LỖI: Vui lòng chỉ định ảnh qua -i/--image, ví dụ:")
            print('  python 02_example_ocr.py -i "path/to/your/image.png"')
            sys.exit(1)

    run_ocr(
        img_path,
        args.output,
        use_latin_model=args.latin,
        engine=args.engine,
        cpu_threads=args.cpu_threads,
    )
