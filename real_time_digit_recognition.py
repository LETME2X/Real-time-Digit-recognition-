import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('handwritten_digit_recognition_model.h5')

# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Canny Edge Detection
    edges = cv2.Canny(thresh, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Recognize digits
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        # Filter contours
        if aspect_ratio > 0.5 and area > 100:
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.reshape((1, 28, 28, 1))
            roi = roi.astype('float32') / 255.0

            pred = model.predict(roi)
            digit = np.argmax(pred)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()