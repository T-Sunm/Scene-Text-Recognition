import os
import tempfile
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

# Load YOLOv11 model once at startup
yolo_model = YOLO("yolo11n.pt")

async def process_image(image_data: bytes) -> Response:
  """Common image processing logic for both URL and file upload"""
  try:
    # Save incoming bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
      temp_file.write(image_data)
      temp_path = temp_file.name

    # Perform detection
    results = yolo_model(temp_path, verbose=False)[0]
    bboxes = results.boxes.xyxy.tolist()
    classes = results.boxes.cls.tolist()
    confs = results.boxes.conf.tolist()
    names = results.names

    # Annotate image
    image = Image.open(temp_path)
    image_array = np.array(image)
    annotator = Annotator(image_array, font="Arial.ttf", pil=False)
    for box, cls, conf in zip(bboxes, classes, confs):
      label = f"{names[int(cls)]} {conf:.2f}"
      annotator.box_label(box, label, color=colors(int(cls), True))

    # Convert back to bytes
    annotated = Image.fromarray(annotator.result())
    buf = BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)

    # Cleanup temp file
    os.unlink(temp_path)
    return Response(content=buf.getvalue(), media_type="image/png")

  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Error processing image: {e}")

@app.get("/detect", response_class=Response)
async def detect_url(image_url: str):
  try:
    resp = requests.get(image_url)
    resp.raise_for_status()
    return await process_image(resp.content)
  except requests.RequestException as e:
    raise HTTPException(
        status_code=400, detail=f"Error downloading image: {e}")

@app.post("/detect/upload", response_class=Response)
async def detect_upload(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")
  content = await file.read()
  return await process_image(content)

# If run directly, use Uvicorn
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
