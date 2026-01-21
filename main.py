import os
import io
import cv2
import numpy as np
import requests
import easyocr
import torch
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# --- 1. SETUP MODELS (CPU ONLY) ---
app = FastAPI(title="PEA Meter Reader API")

# EasyOCR (CPU, Silent)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# YOLO Model
yolo_model = None
try:
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
    else:
        print(f"❌ Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")

# Session Setup
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- 2. DATA MODELS (Validation Layer) ---
class ImageInput(BaseModel):
    url: str

    @field_validator('url')
    def validate_url(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('URL cannot be empty')
        if not v.lower().startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

# --- 3. HELPER FUNCTIONS ---
def preprocess_image(img_cv2):
    """ปรับภาพให้ชัดขึ้นก่อนอ่าน OCR"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        img_gray = ImageOps.grayscale(img_pil)
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(2.0)
        return np.array(img_enhanced)
    except Exception:
        return img_cv2

def read_text_from_crop(img_crop):
    """อ่านค่าจากภาพที่ Crop มาแล้ว"""
    enhanced_crop = preprocess_image(img_crop)
    
    # 1. Barcode
    barcodes = decode(enhanced_crop)
    if barcodes:
        valid_barcodes = []
        for b in barcodes:
            txt = b.data.decode('utf-8')
            if 5 < len(txt) <= 15: # ปรับ Range ให้เหมาะสม
                valid_barcodes.append(txt)
        if valid_barcodes:
            return valid_barcodes[0], "barcode"

    # 2. OCR
    ocr_results = reader.readtext(enhanced_crop, detail=0, allowlist='0123456789')
    valid_numbers = [num for num in ocr_results if 5 < len(num) <= 12]
    
    if valid_numbers:
        return valid_numbers[0], "ocr"
        
    return None, None

# --- 4. API ENDPOINT (The Pipeline) ---
@app.post("/predict")
def process_meter_reading(input_data: ImageInput):
    """
    Pipeline: Validation -> Download -> Decode -> Predict -> Result
    """
    target_url = input_data.url
    
    # --- STEP 1: Download Image ---
    try:
        response = session.get(target_url, timeout=10)
        response.raise_for_status() # เช็ค HTTP Error (404, 500)
        image_bytes = io.BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=400, content={
            "status": "failed",
            "step": "download_image",
            "message": f"Network error: {str(e)}",
            "url": target_url
        })

    # --- STEP 2: Decode Image (Opencv) ---
    try:
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_np is None:
            raise ValueError("File is not a valid image")
            
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "status": "failed",
            "step": "process_image",
            "message": "Invalid image file format",
            "url": target_url
        })

    # --- STEP 3: Model Prediction (YOLO) ---
    if yolo_model is None:
        return JSONResponse(status_code=500, content={
            "status": "failed",
            "step": "model_loading",
            "message": "AI Model not loaded on server"
        })

    try:
        results = yolo_model(img_np, verbose=False)
        
        found_serial = None
        found_method = None
        
        # วนลูปหา Box ที่ดีที่สุด
        for result in results:
            boxes = result.boxes
            # เรียงตามความมั่นใจ
            sorted_indices = torch.argsort(boxes.conf, descending=True)

            for i in sorted_indices:
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Crop ภาพ
                h, w = img_np.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop_img = img_np[y1:y2, x1:x2]
                
                if crop_img.size == 0: continue

                # อ่านค่า (OCR/Barcode)
                serial, method = read_text_from_crop(crop_img)
                
                if serial:
                    found_serial = serial
                    found_method = method
                    break # เจอแล้วหยุดเลย
            
            if found_serial: break

        # --- STEP 4: Final Result Logic ---
        if found_serial:
            return {
                "status": "success",
                "step": "finished",
                "data": {
                    "serial_number": found_serial,
                    "method": found_method,
                    "confidence": "high" # หรือใส่ค่า conf จริงถ้าต้องการ
                }
            }
        else:
            return JSONResponse(status_code=200, content={
                "status": "failed",
                "step": "prediction_result",
                "message": "Object detected but no readable serial number found",
                "data": None
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "failed",
            "step": "model_prediction",
            "message": f"Internal AI Error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)