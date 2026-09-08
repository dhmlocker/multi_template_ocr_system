from pathlib import Path
import os

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_DEV_MODEL_WL", "1")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DET_DIR = MODEL_DIR / "PP-OCRv6_medium_det"
REC_DIR = MODEL_DIR / "PP-OCRv6_medium_rec"
SAMPLE = ROOT / "data" / "generated" / "images" / "reimbursement" / "reimbursement_0000_clear.png"

print("[1/2] 下载 MobileNetV2...")
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
print("MobileNetV2 完成")

print("[2/2] 下载 PaddleOCR 检测和识别模型...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang="ch",
    device="cpu",
    engine="paddle",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_dir=str(DET_DIR),
    text_recognition_model_dir=str(REC_DIR),
)
if SAMPLE.exists():
    list(ocr.predict(str(SAMPLE)))
print("PaddleOCR 完成")
print(f"模型目录：{MODEL_DIR}")
