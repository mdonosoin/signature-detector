import os
import numpy as np
import cv2
import fitz  # PyMuPDF
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Load model
REPO_DIR = Path(__file__).resolve().parent
LOCAL_MODEL = REPO_DIR / "yolov8s.pt"

MODEL_PATH = os.environ.get("MODEL_PATH")

if MODEL_PATH:
    model_path = MODEL_PATH
elif LOCAL_MODEL.exists():
    model_path = str(LOCAL_MODEL)
else:
    model_path = hf_hub_download(
        repo_id="tech4humans/yolov8s-signature-detector", filename="yolov8s.pt"
    )

model = YOLO(model_path)


def _pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200):
    """
    Convierte un PDF (en bytes) a una lista de imágenes OpenCV (BGR) sin usar Poppler.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    # factor DPI -> matriz
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        # pix.n suele ser 3 (RGB). Convertimos a BGR para OpenCV/consistencia
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    doc.close()
    return images


def detectar_firmas_en_imagen(image_bgr: np.ndarray) -> int:
    """
    Recibe imagen OpenCV (BGR) y devuelve número de detecciones.
    """
    results = model(image_bgr)
    boxes = results[0].boxes
    if boxes is None:
        return 0
    return int(len(boxes))


def analizar_pdf_bytes(pdf_bytes: bytes, dpi: int = 200):
    """
    Pipeline completo: PDF bytes -> imágenes -> detección por página.
    Devuelve JSON serializable.
    """
    images = _pdf_bytes_to_images(pdf_bytes, dpi=dpi)

    paginas = []
    for idx, img in enumerate(images):
        num_firmas = detectar_firmas_en_imagen(img)
        paginas.append({"pagina": idx + 1, "firmas_detectadas": int(num_firmas)})

    firma_ok = any(p["firmas_detectadas"] > 0 for p in paginas)

    return {"firma_OK": firma_ok, "paginas": paginas}
