import cv2
import subprocess
import time
from detect_plate import detect_plate

# Open default camera (index 0 is more common on Windows)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to access the camera.")
    exit()

print("✅ Camera opened successfully. Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(10)
    if key != -1:
        print(f"🔑 Key pressed: {key}")  # Debug output

    if key == ord('s'):
        timestamp = int(time.time())
        image_path = f"captured_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"📸 Saved frame as {image_path}")

        # Call another Python script with image path
        detect_plate(image_path)
        #subprocess.run(["python", "detect_plate.py", image_path])

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
