import cv2

cap = cv2.VideoCapture(1)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Camera not accessible")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured")
            break

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()