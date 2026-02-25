import os
import uuid
import cv2
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 1️⃣ Cargar modelo una sola vez (muy importante)
model_path = hf_hub_download(
    repo_id="tech4humans/yolov8s-signature-detector", filename="yolov8s.pt"
)

model = YOLO(model_path)

POPPLER_BIN = (
    r"C:\Users\m.donosoin\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
)


# 2️⃣ Convertir PDF a imágenes
def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_BIN)
    image_paths = []

    for i, page in enumerate(pages):
        image_name = f"{uuid.uuid4()}_page_{i}.png"
        page.save(image_name, "PNG")
        image_paths.append(image_name)

    return image_paths


# 3️⃣ Analizar una imagen
def detectar_firmas_en_imagen(image_path):
    results = model(image_path)

    # Extraemos detecciones
    detections = results[0].boxes

    if detections is None:
        return 0

    return len(detections)


# 4️⃣ Pipeline completo PDF
def analizar_pdf(pdf_path):

    image_paths = pdf_to_images(pdf_path)

    resultado_paginas = []

    for idx, image_path in enumerate(image_paths):
        num_firmas = detectar_firmas_en_imagen(image_path)

        resultado_paginas.append(
            {"pagina": idx + 1, "firmas_detectadas": int(num_firmas)}
        )

        # Borramos imagen temporal
        os.remove(image_path)

    documento_valido = any(p["firmas_detectadas"] > 0 for p in resultado_paginas)

    return {"firma_OK": documento_valido, "paginas": resultado_paginas}


resultado = analizar_pdf(
    "C:/Users/m.donosoin/Desktop/signature-detector/data/firma_electronica.pdf"
)
print(resultado)
