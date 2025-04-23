from fastai.vision.all import *
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO

learn = load_learner('pneumonia_model.pkl')
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = PILImage.create(BytesIO(img_bytes))
    pred, idx, probs = learn.predict(img)
    return JSONResponse({
        "prediction": pred,
        "confidence": f"{probs[idx]*100:.2f}%"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

