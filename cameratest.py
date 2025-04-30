import cv2

# Try opening /dev/video0 (default USB camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to access the camera.")
    exit()

print("✅ Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Jetson Nano Camera Feed", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

