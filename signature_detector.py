import os
import gc
import numpy as np
import cv2
import fitz  # PyMuPDF
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==============================
# CARGA MODELO (una sola vez)
# ==============================

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


# ==============================
# DETECCIÓN EN IMAGEN
# ==============================


def detectar_firmas_en_imagen(image_bgr: np.ndarray) -> int:
    results = model(image_bgr)
    boxes = results[0].boxes
    if boxes is None:
        return 0
    return int(len(boxes))


# ==============================
# PROCESAMIENTO OPTIMIZADO PDF
# ==============================


def analizar_pdf_bytes(pdf_bytes: bytes, dpi: int = 100):
    """
    Procesa el PDF página a página sin almacenar todas las imágenes.
    Optimizado para entornos de 512MB.
    """

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    paginas = []
    firma_ok = False

    for idx, page in enumerate(doc):

        # Convertir página a pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convertir a numpy directamente
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detectar
        num_firmas = detectar_firmas_en_imagen(img)

        paginas.append({"pagina": idx + 1, "firmas_detectadas": int(num_firmas)})

        if num_firmas > 0:
            firma_ok = True

        # LIBERAR MEMORIA
        del pix
        del img
        gc.collect()

    doc.close()
    gc.collect()

    return {"firma_OK": firma_ok, "paginas": paginas}
