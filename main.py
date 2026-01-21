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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Tuple, Optional, Any

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# --- 1. SETUP SYSTEM ---
app = FastAPI(title="PEA Meter Reader API")

# EasyOCR (CPU, Silent)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Load YOLO
def load_yolo_model(path: str):
    try:
        if os.path.exists(path):
            return YOLO(path)
        print(f"❌ Warning: Model file not found at {path}")
        return None
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return None

yolo_model = load_yolo_model(MODEL_PATH)

# Session Setup
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- 2. DATA MODELS ---
class ImageInput(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('URL cannot be empty')
        if not v.lower().startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

# --- 3. HELPER FUNCTIONS (Extracted to reduce complexity) ---

def preprocess_image(img_cv2: np.ndarray) -> np.ndarray:
    """ปรับแต่งภาพให้ Contrast จัดขึ้น"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        img_gray = ImageOps.grayscale(img_pil)
        enhancer = ImageEnhance.Contrast(img_gray)
        return np.array(enhancer.enhance(2.0))
    except Exception:
        return img_cv2

def read_text_from_crop(img_crop: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """อ่านค่า Barcode หรือ OCR จากภาพที่ Crop แล้ว"""
    enhanced_crop = preprocess_image(img_crop)
    
    # 1. Try Barcode
    barcodes = decode(enhanced_crop)
    if barcodes:
        for b in barcodes:
            txt = b.data.decode('utf-8')
            if 5 < len(txt) <= 15:
                return txt, "barcode"

    # 2. Try OCR
    ocr_results = reader.readtext(enhanced_crop, detail=0, allowlist='0123456789')
    valid_numbers = [num for num in ocr_results if 5 < len(num) <= 12]
    
    if valid_numbers:
        return valid_numbers[0], "ocr"
        
    return None, None

def download_image_from_url(url: str) -> bytes:
    """ฟังก์ชันแยกสำหรับการดาวน์โหลด"""
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.content

def decode_image_bytes(image_content: bytes) -> np.ndarray:
    """ฟังก์ชันแยกสำหรับการแปลง Bytes เป็น OpenCV Image"""
    file_bytes = np.asarray(bytearray(image_content), dtype=np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_np is None:
        raise ValueError("Decoded image is None")
    return img_np

def detect_and_read_meter(img_np: np.ndarray, model: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Logic หลักที่ซับซ้อนถูกแยกมาไว้ที่นี่ (Reducing Cognitive Complexity)
    ทำหน้าที่: YOLO Detect -> Loop Boxes -> Crop -> OCR -> Return Result
    """
    results = model(img_np, verbose=False)
    
    for result in results:
        # เรียงลำดับความมั่นใจจากมากไปน้อย
        if not hasattr(result, 'boxes'):
            continue
            
        boxes = result.boxes
        sorted_indices = torch.argsort(boxes.conf, descending=True)

        for i in sorted_indices:
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Check crop boundaries
            h, w = img_np.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = img_np[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            # อ่านค่า
            serial, method = read_text_from_crop(crop_img)
            if serial:
                return serial, method # เจอแล้ว Return ทันที (Early Return)
                
    return None, None

# --- 4. API ENDPOINT (Orchestrator) ---
@app.post("/predict")
def process_meter_reading(input_data: ImageInput):
    """
    Main Pipeline: อ่านง่าย เป็นลำดับขั้นตอน ไม่ซ้อน Loop ลึก
    """
    target_url = input_data.url
    
    # --- STEP 1: Download ---
    try:
        image_content = download_image_from_url(target_url)
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "status": "failed", "step": "download_image",
            "message": str(e), "url": target_url
        })

    # --- STEP 2: Decode ---
    try:
        img_np = decode_image_bytes(image_content)
    except Exception:
        return JSONResponse(status_code=400, content={
            "status": "failed", "step": "process_image",
            "message": "Invalid image format", "url": target_url
        })

    # --- STEP 3: Validate Model ---
    if yolo_model is None:
        return JSONResponse(status_code=500, content={
            "status": "failed", "step": "model_loading",
            "message": "AI Model not loaded"
        })

    # --- STEP 4: Predict & Read (Logic แยกออกไปแล้ว) ---
    try:
        serial, method = detect_and_read_meter(img_np, yolo_model)
        
        if serial:
            return {
                "status": "success",
                "step": "finished",
                "data": {"serial_number": serial, "method": method}
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
            "status": "failed", "step": "model_prediction",
            "message": f"AI Error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)