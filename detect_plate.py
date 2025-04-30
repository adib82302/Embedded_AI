import cv2
import pytesseract
from ultralytics import YOLO

# Load image and model
image_path = "test_image.jpg"
model = YOLO("best.pt")
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image could not be loaded.")

# Detect plates
results = model(image_path)[0]

for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_crop = img[y1:y2, x1:x2]

    # OCR preprocessing
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Run OCR
    text = pytesseract.image_to_string(cleaned, config='--psm 7')
    text = text.strip()

    print(f"[Plate {i+1}] OCR Result: '{text}'")

    if len(text) < 4:
        print("Plate may be obstructed or unreadable.")
