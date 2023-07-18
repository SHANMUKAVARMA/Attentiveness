import cv2
import dlib

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Open the default webcam
cap = cv2.VideoCapture(0)

while True:
    def eye_aspect_ratio(eye):
        # Calculate the vertical distances between the eye landmarks
        a = distance(eye[1], eye[5])
        b = distance(eye[2], eye[4])

        # Calculate the horizontal distance between the eye corners
        c = distance(eye[0], eye[3])

        # Calculate the eye aspect ratio (EAR)
        ear = (a + b) / (2.0 * c)

        return ear

    def distance(p1, p2):
        # Calculate the Euclidean distance between two points
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each face detected
    for face in faces:
        # Predict the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Extract the eye landmarks
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Calculate the eye aspect ratio (EAR) for each eye
        left_ear = eye_aspect_ratio([landmarks.part(36), landmarks.part(37), landmarks.part(38), landmarks.part(39), landmarks.part(40), landmarks.part(41)])
        right_ear = eye_aspect_ratio([landmarks.part(42), landmarks.part(43), landmarks.part(44), landmarks.part(45), landmarks.part(46), landmarks.part(47)])

        # Calculate the average EAR
        ear = (left_ear + right_ear) / 2.0

        # Determine if the student is attentive or not based on the EAR
        if ear < 0.20:
            cv2.putText(frame, "Not Attentive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Attentive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the eye landmarks on the frame for visualization
        cv2.circle(frame, (left_eye.x, left_eye.y), 1, (0, 0, 255), -1)
        cv2.circle(frame, (right_eye.x, right_eye.y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
