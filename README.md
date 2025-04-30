# License Plate Detection & OCR

A compact pipeline using YOLOv8 + Tesseract to detect and read license plates.

## Dependencies

- **Python 3.11**  
- **Tesseract OCR**  
  - Ubuntu/Debian: `sudo apt install -y tesseract-ocr libtesseract-dev`  
  - macOS (Homebrew): `brew install tesseract`

## Setup

Ensure Tesseract binary is installed

Ubuntu/Debian specific install
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```
Create virtual-env
```bash
python3.11 -m venv plate_env && source plate_env/bin/activate
pip install --upgrade pip
pip install \
  torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
  ultralytics opencv-python-headless pytesseract numpy==1.26.4
```

## Yolov8 Training

Download license plate dataset [here](https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk/dataset/2/download)

Create `data.yaml` with the following structure:
```yaml
path: /path to image set

train: train/images
val: valid/images

nc: 1
names: ['license_plate']
```

Train Yolov8n model on license plate dataset
```
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=5
```
