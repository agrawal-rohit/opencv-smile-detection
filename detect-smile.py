import cv2

# Load the Haar cascades xml files for face, eye and smile detection
face = cv2.CascadeClassifier("haar-features/frontal-face.xml")
eye = cv2.CascadeClassifier("haar-features/eye.xml")
smile = cv2.CascadeClassifier("haar-features/smile.xml")


# Function to detect smile in a given frame
def detect_smile(gray, frame):
    # Detect faces in the grayscale image
    faces = face.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define regions of interest in the grayscale and colored images
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detect eyes in the grayscale ROI
        eyes = eye.detectMultiScale(roi_gray, 1.1, 8)

        # Detect smiles in the grayscale ROI
        smiles = smile.detectMultiScale(roi_gray, 2, 22)

        # Initialize smile_found to False
        smile_found = False

        # Draw rectangles around detected eyes
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Draw rectangles around detected smiles and set smile_found to True
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            smile_found = True

        # If no smile is found, display "Neutral", else display "Happy"
        if smile_found is False:
            cv2.putText(
                frame,
                "Neutral",
                (x + 10, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Happy",
                (x + 10, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                2,
            )

    # Return the frame with detected faces, eyes, and smiles
    return frame


# Start video capture
video_capture = cv2.VideoCapture(0)

# Main loop for video capture and smile detection
while True:
    # Read a frame from the video capture
    _, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the grayscale frame
    canvas = detect_smile(gray, frame)

    # Display the frame with detected smiles
    cv2.imshow("Video", canvas)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
