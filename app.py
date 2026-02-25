from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from signature_detector import analizar_pdf_bytes

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


# Mantengo /predict (porque ya lo tienes) y añado /detect como alias
@app.post("/predict")
@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        resultado = analizar_pdf_bytes(pdf_bytes, dpi=200)
        return resultado
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
