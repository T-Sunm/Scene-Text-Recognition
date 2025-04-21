import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from crnn import CRNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

# Constants
TEXT_DET_MODEL_PATH = "../model/yolo11m/best.pt"
OCR_MODEL_PATH = "../ocr_crnn.pt"

# Character set configuration
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# Model configuration
HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3

# Load YOLO detection model
det_model = YOLO(TEXT_DET_MODEL_PATH)

# Initialize CRNN recognition model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg_model = CRNN(
    vocab_size=len(CHARS),
    hidden_size=HIDDEN_SIZE,
    n_layers=N_LAYERS,
    dropout=DROPOUT_PROB,
    unfreeze_layers=UNFREEZE_LAYERS,
).to(device)
reg_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=device))
reg_model.eval()

# Preprocessing transform for CRNN
transform = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Utility to decode CTC outputs
def ctc_decode(seq_tokens):
  decoded = []
  prev = None
  for t in seq_tokens:
    if t != 0 and t != prev:
      decoded.append(IDX_TO_CHAR[t])
    prev = t
  return "".join(decoded)

# Endpoint logic: detection
async def annotate_detection(image_data: bytes) -> Response:
  try:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
      tmp.write(image_data)
      path = tmp.name

    # YOLO detection
    res = det_model(path, verbose=False)[0]
    bboxes = res.boxes.xyxy.tolist()
    classes = res.boxes.cls.tolist()
    confs = res.boxes.conf.tolist()
    names = res.names

    # Load and annotate image
    img = Image.open(path)
    arr = np.array(img)
    annotator = Annotator(arr, font="Arial.ttf", pil=False)
    for box, cls, conf in zip(bboxes, classes, confs):
      label = f"{names[int(cls)]} {conf:.2f}"
      annotator.box_label(box, label, color=colors(int(cls), True))

    # Return annotated image
    out = Image.fromarray(annotator.result())
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    os.unlink(path)
    return Response(content=buf.getvalue(), media_type="image/png")

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# Endpoint logic: OCR
async def annotate_ocr(image_data: bytes) -> Response:
  try:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
      tmp.write(image_data)
      path = tmp.name

    # Detect text regions
    res = det_model(path, verbose=False)[0]
    bboxes = res.boxes.xyxy.tolist()
    classes = res.boxes.cls.tolist()
    confs = res.boxes.conf.tolist()
    names = res.names

    # Load image and prepare annotator
    img = Image.open(path)
    arr = np.array(img)
    annotator = Annotator(arr, font="Arial.ttf", pil=False)

    # Recognize and annotate each region
    for box, cls, conf in zip(bboxes, classes, confs):
      x1, y1, x2, y2 = map(int, box)
      crop = img.crop((x1, y1, x2, y2))
      tensor = transform(crop).unsqueeze(0).to(device)
      with torch.no_grad():
        logits = reg_model(tensor).cpu()
      seq = logits.permute(1, 0, 2).argmax(2)[0].tolist()
      text = ctc_decode(seq)
      label = f"{names[int(cls)][:3]}{conf:.1f}:{text}"
      annotator.box_label([x1, y1, x2, y2], label,
                          color=colors(int(cls), True))

    # Return annotated OCR image
    out = Image.fromarray(annotator.result())
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    os.unlink(path)
    return Response(content=buf.getvalue(), media_type="image/png")

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoints
@app.get("/detect", response_class=Response)
async def detect_url(image_url: str):
  try:
    r = requests.get(image_url)
    r.raise_for_status()
    return await annotate_detection(r.content)
  except requests.RequestException as e:
    raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect/upload", response_class=Response)
async def detect_upload(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")
  data = await file.read()
  return await annotate_detection(data)

@app.get("/ocr", response_class=Response)
async def ocr_url(image_url: str):
  try:
    r = requests.get(image_url)
    r.raise_for_status()
    return await annotate_ocr(r.content)
  except requests.RequestException as e:
    raise HTTPException(status_code=400, detail=str(e))

@app.post("/ocr/upload", response_class=Response)
async def ocr_upload(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")
  data = await file.read()
  return await annotate_ocr(data)

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
