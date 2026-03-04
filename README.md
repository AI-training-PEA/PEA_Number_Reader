# ตัวอย่าง API ที่ใช้เรียก

```bash
    curl --request POST \
    --url http://127.0.0.1:8000/predict \
    --header 'content-type: application/json' \
    --data '{
    "url": "https://webservice.pea.co.th/SurveyImage/BPNL/202601_BPNL_219_A3_BPNL0005_6601379771_020009421021.jpg"
    }'
```

# Workflow
<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b> คลิกเพื่อดู Workflow </b></span></summary>

## 1.function download_image_from_url

```python
    def download_image_from_url(url: str) -> bytes:
        """ฟังก์ชันแยกสำหรับการดาวน์โหลด"""
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.content
```

### input :

```
"https://webservice.pea.co.th/SurveyImage/BPNL/202601_BPNL_219_A3_BPNL0005_6601379771_020009421021.jpg"
```

### output :
<details>
<summary><b> คลิกเพื่อดูตัวอย่าง Output (ข้อมูลดิบ Raw Bytes) </b></summary>

```text
b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x00\x00\x00\x00\x00\xff\xe2\x02(ICC_PROFILE\x00\x01\x01\x00\x00\x02\x18\x00\x00\x00\x00\x040\x00\x00mntrRGB XYZ ... )

```

</details>

## 2.function decode_image_bytes

```python
def decode_image_bytes(image_content: bytes) -> np.ndarray:
    """ฟังก์ชันแยกสำหรับการแปลง Bytes เป็น OpenCV Image"""
    file_bytes = np.asarray(bytearray(image_content), dtype=np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_np is None:
        raise ValueError("Decoded image is None")
    return img_np
```

### input : image
<details>
<summary><b> คลิกเพื่อดู Input (ข้อมูลดิบ Raw Bytes) </b></summary>

```text
b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x00\x00\x00\x00\x00\xff\xe2\x02(ICC_PROFILE\x00\x01\x01\x00\x00\x02\x18\x00\x00\x00\x00\x040\x00\x00mntrRGB XYZ ... )
```

</details>

### output:
<details>
<summary><b> คลิกเพื่อดู Output (ข้อมูล Matrix)</b></summary>

```text
[[[ 59  39  34]
  [ 64  44  39]
  [ 69  47  41]
  ...
  [254 254 254]
  [254 254 254]
  [254 254 254]]

 [[ 58  38  33]
  [ 63  43  38]
  [ 69  47  42]
  ...
  [254 254 254]
  [254 254 254]
  [254 254 254]]

 [[ 61  42  35]
  [ 62  42  37]
  [ 70  47  45]
  ...
  [253 253 253]
  [254 254 254]
  [254 254 254]]

 ...

 [[ 29  30  56]
  [ 33  34  78]
  [ 31  32 113]
  ...
  [ 39  43  54]
  [ 33  37  48]
  [ 36  39  53]]

 [[ 32  34  42]
  [ 27  28  42]
  [ 16  18  48]
  ...
  [ 50  52  63]
  [ 46  48  59]
  [ 46  47  61]]

 [[ 27  30  28]
  [ 24  26  26]
  [ 22  25  29]
  ...
  [ 55  57  68]
  [ 48  50  61]
  [ 47  47  61]]]
```

</details>

## 3.function detect_and_read_meter

```python
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
```

### Input : image
<details>
<summary><b> คลิกเพื่อดู Output (ข้อมูล Matrix)</b></summary>

```text
[[[ 59  39  34]
  [ 64  44  39]
  [ 69  47  41]
  ...
  [254 254 254]
  [254 254 254]
  [254 254 254]]

 [[ 58  38  33]
  [ 63  43  38]
  [ 69  47  42]
  ...
  [254 254 254]
  [254 254 254]
  [254 254 254]]

 [[ 61  42  35]
  [ 62  42  37]
  [ 70  47  45]
  ...
  [253 253 253]
  [254 254 254]
  [254 254 254]]

 ...

 [[ 29  30  56]
  [ 33  34  78]
  [ 31  32 113]
  ...
  [ 39  43  54]
  [ 33  37  48]
  [ 36  39  53]]

 [[ 32  34  42]
  [ 27  28  42]
  [ 16  18  48]
  ...
  [ 50  52  63]
  [ 46  48  59]
  [ 46  47  61]]

 [[ 27  30  28]
  [ 24  26  26]
  [ 22  25  29]
  ...
  [ 55  57  68]
  [ 48  50  61]
  [ 47  47  61]]]
```
</details>

### Input : Model
<details>
<summary><b> คลิกเพื่อดู Input (ข้อมูลModel) </b></summary>

```text
YOLO(
  (model): DetectionModel(
    (model): Sequential(
      (0): Conv(
        (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (1): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (2): C3k2(
        (cv1): Conv(
          (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (4): C3k2(
        (cv1): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (5): Conv(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (6): C3k2(
        (cv1): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (7): Conv(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (8): C3k2(
        (cv1): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (9): SPPF(
        (cv1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      )
      (10): C2PSA(
        (cv1): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): Sequential(
          (0): PSABlock(
            (attn): Attention(
              (qkv): Conv(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (ffn): Sequential(
              (0): Conv(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
          (1): PSABlock(
            (attn): Attention(
              (qkv): Conv(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (proj): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
              (pe): Conv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
            (ffn): Sequential(
              (0): Conv(
                (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): Identity()
              )
            )
          )
        )
      )
      (11): Upsample(scale_factor=2.0, mode='nearest')
      (12): Concat()
      (13): C3k2(
        (cv1): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (14): Upsample(scale_factor=2.0, mode='nearest')
      (15): Concat()
      (16): C3k2(
        (cv1): Conv(
          (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (17): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (18): Concat()
      (19): C3k2(
        (cv1): Conv(
          (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (20): Conv(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (21): Concat()
      (22): C3k2(
        (cv1): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (cv2): Conv(
          (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (m): ModuleList(
          (0-1): 2 x C3k(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv3): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): Sequential(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
              (1): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
        )
      )
      (23): Detect(
        (cv2): ModuleList(
          (0): Sequential(
            (0): Conv(
              (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (1-2): 2 x Sequential(
            (0): Conv(
              (conv): Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (cv3): ModuleList(
          (0): Sequential(
            (0): Sequential(
              (0): DWConv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): Sequential(
              (0): DWConv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
          )
          (1-2): 2 x Sequential(
            (0): Sequential(
              (0): DWConv(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
                (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (1): Sequential(
              (0): DWConv(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
              (1): Conv(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                (act): SiLU(inplace=True)
              )
            )
            (2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (dfl): DFL(
          (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )
    )
  )
)

```

</details>

### Output : Tuple
```
('6601379771', 'ocr')
```

## 4.function process_meter_reading

```python
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
```

### Input : ImageInput
```
url='https://webservice.pea.co.th/SurveyImage/BPNL/202601_BPNL_219_A3_BPNL0005_6601379771_020009421021.jpg'

```

### Output : JSON / Dict
```
{'status': 'success', 'step': 'finished', 'data': {'serial_number': '6601379771', 'method': 'ocr'}}
```

</details>
<br>

# อธิบายฟังก์ชั่นต่างๆ

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>process_meter_reading</b></span></summary>

เป็น API Endpoint หลัก (/predict) ที่ควบคุมลำดับการทำงาน (Pipeline) ทั้งหมด ทำหน้าที่รับ Request เข้ามา สั่งการฟังก์ชันอื่นๆ ตามลำดับขั้นตอน และรวบรวมผลลัพธ์เพื่อส่งกลับ (Response) ในรูปแบบ JSON

```python
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
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>validate_url</b></span></summary>

เป็นฟังก์ชันตรวจสอบ (Validator) ของ Pydantic ใช้สำหรับกรองข้อมูล URL ที่ผู้ใช้อัปโหลดเข้ามา โดยบังคับว่าต้องไม่เป็นค่าว่าง และต้องขึ้นต้นด้วย http:// หรือ https:// เท่านั้น เพื่อป้องกัน Error ในขั้นตอนดาวน์โหลด

```python
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('URL cannot be empty')
        if not v.lower().startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>download_image_from_url</b></span></summary>

รับค่า URL และทำการดาวน์โหลดรูปภาพผ่าน HTTP Request โดยมีการตั้งค่า Timeout และคืนค่ากลับมาเป็นข้อมูลไฟล์ภาพดิบ (Bytes)

```python
def download_image_from_url(url: str) -> bytes:
    """ฟังก์ชันแยกสำหรับการดาวน์โหลด"""
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.content
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>decode_image_bytes</b></span></summary>

นำข้อมูล Bytes ที่ได้จากการดาวน์โหลด มาแปลง (Decode) ให้เป็นโครงสร้างข้อมูล Numpy Array เพื่อให้ไลบรารี OpenCV สามารถนำรูปภาพนี้ไปประมวลผลต่อได้

```python
def decode_image_bytes(image_content: bytes) -> np.ndarray:
    """ฟังก์ชันแยกสำหรับการแปลง Bytes เป็น OpenCV Image"""
    file_bytes = np.asarray(bytearray(image_content), dtype=np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_np is None:
        raise ValueError("Decoded image is None")
    return img_np
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>preprocess_image</b></span></summary>

ปรับปรุงคุณภาพของรูปภาพ (Image Enhancement) ก่อนส่งเข้ากระบวนการสกัดอักษร โดยจะแปลงภาพให้เป็นสีขาวดำ (Grayscale) และเร่งความเปรียบต่าง (Contrast) เพื่อลดจุดรบกวนและทำให้ตัวเลขคมชัดขึ้น

```python
def preprocess_image(img_cv2: np.ndarray) -> np.ndarray:
    """ปรับแต่งภาพให้ Contrast จัดขึ้น"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        img_gray = ImageOps.grayscale(img_pil)
        enhancer = ImageEnhance.Contrast(img_gray)
        return np.array(enhancer.enhance(2.0))
    except Exception:
        return img_cv2
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>load_yolo_model</b></span></summary>

ทำการโหลดไฟล์โมเดล YOLO (.pt) เข้าสู่หน่วยความจำของระบบตั้งแต่ตอนเริ่มรันเซิร์ฟเวอร์ พร้อมระบบจัดการ Error (Exception Handling) เพื่อแจ้งเตือนหากไฟล์โมเดลสูญหาย

```python
def load_yolo_model(path: str):
    try:
        if os.path.exists(path):
            return YOLO(path)
        print(f"❌ Warning: Model file not found at {path}")
        return None
    except Exception as e:
        print(f"❌ Error loading YOLO: {e}")
        return None
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>detect_and_read_meter</b></span></summary>

นำรูปภาพเต็มส่งเข้าโมเดล YOLO เพื่อค้นหาพิกัด (Bounding Box) ของหน้าปัดมิเตอร์ จากนั้นเรียงลำดับผลลัพธ์ตามค่าความมั่นใจ (Confidence) ทำการตัดภาพ (Crop) เฉพาะส่วนหน้าปัด และส่งไปอ่านค่าตัวเลข

```python
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
```

</details>

<details>
<summary><span style="font-size: 1.5em; color: #c1dfff;"><b>read_text_from_crop</b></span></summary>

ถอดรหัสตัวเลขจากภาพที่ถูกตัดมา (Crop) โดยใช้กลยุทธ์แบบ 2 ขั้นตอน คือ พยายามอ่านจาก Barcode ก่อนเนื่องจากมีความแม่นยำสูง หากไม่สำเร็จจึงจะสลับไปใช้ EasyOCR ในการสกัดตัวอักษรแทน

```python
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
```

</details>

