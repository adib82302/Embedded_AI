import sys
import os
import cv2
import pytesseract
import argparse
from ultralytics import YOLO

def detect_plate(image_path):
    output_file = open('outputs/output.txt', 'w')

    # Redirect stdout to the file
    sys.stdout = output_file

    # Load image and model
    #image_path = "test_images/3069000.jpg"
    OUT_DIR   = "outputs"

    # make sure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    print("load yolov8 model")
    model = YOLO("runs/detect/train/weights/best.pt")
    print("model loaded.")
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image could not be loaded.")

    # Detect plates
    print("detecting license plate with yolov8 model")
    results = model(image_path)[0]

    annotated = img.copy()

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        plate_crop = img[y1:y2, x1:x2]

        # 1) crop
        plate_crop = img[y1:y2, x1:x2]

        # 2) annotate full image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            annotated,
            f"{i+1}:{conf:.2f}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2,
            cv2.LINE_AA
        )

        # OCR preprocessing
        # 1) grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # 2) slight Gaussian blur to smooth noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 3) Otsu's threshold to get a clean binary image
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # clean noise with morphology to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # save cropped preprocessed image
        crop_path = os.path.join(OUT_DIR, f"plate_{i+1}.jpg")
        cv2.imwrite(crop_path, cleaned)
        print(f"Saved crop → {crop_path} (conf={conf:.2f})")
        
        # Run OCR
        text = pytesseract.image_to_string(cleaned, config='--psm 7')
        text = text.strip()

        print(f"[Plate {i+1}] OCR Result: '{text}'")

        if len(text) < 4:
            print("Plate may be obstructed or unreadable.")

    # save the annotated image
    annot_path = os.path.join(OUT_DIR, "annotated.jpg")
    cv2.imwrite(annot_path, annotated)
    print(f"Saved annotated image → {annot_path}")

    output_file.close()
    sys.stdout = sys.__stdout__  # Reset stdout to default
    print("Output saved to outputs/output.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect license plate from image.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()
    image_path = args.image_path
    detect_plate(image_path)
