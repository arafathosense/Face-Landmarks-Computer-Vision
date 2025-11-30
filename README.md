# Face Landmarks Detection using MediaPipe and OpenCV

This repository provides a real-time facial landmark detection system built using MediaPipe Face Mesh and OpenCV. The model identifies 468 high-precision facial landmarks, enabling detailed analysis of facial geometry, contours, eye regions, and mouth structure for various computer vision applications.

# Features

    Detection of 468 facial landmarks with high accuracy
    
    Real-time processing using a standard webcam
    
    Robust tracking of facial contours, eyes, lips, and mesh topology
    
    Lightweight and optimized for high-performance applications
    
    Easy to extend for advanced face-analysis tasks

#  Technologies Used

    Python 3.x
    
    OpenCV
    
    MediaPipe (Face Mesh Solution)
    
    NumPy (optional for further processing)

# Installation

Install required packages:

    pip install opencv-python mediapipe numpy


# Output
<img width="1365" height="725" alt="image" src="https://github.com/user-attachments/assets/5ae237c5-0d35-40cf-beff-24a87eb947ee" />
<img width="1365" height="724" alt="image" src="https://github.com/user-attachments/assets/13e7b586-57af-4816-9ed6-6aa343f08770" />
<img width="1365" height="724" alt="image" src="https://github.com/user-attachments/assets/30ddfb1d-12ab-4042-8367-df6d69c1aca9" />



# Code Overview
    import cv2
    import mediapipe as mp
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4)
    mp_drawing = mp.solutions.drawing_utils
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Convert color format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
    
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for point in landmarks.landmark:
                    x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
                # mp_drawing.draw_landmarks(
                #     frame,
                #     landmarks,
                #     mp_face_mesh.FACEMESH_TESSELATION,  # Facial connections
                #     landmark_drawing_spec=None,  # Default landmark style
                #     # connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                # )
    
        cv2.imshow('Face Landmarks', frame)
        cv2.waitKey(1)
    
        if cv2.getWindowProperty('Face Landmarks', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Applications

    Facial geometry analysis
    
    Real-time AR filter development
    
    Eye, mouth, and facial movement tracking
    
    Emotion and expression analysis
    
    Head pose estimation research    

# Contribution

Contributions, suggestions, and improvements are welcome.
Please feel free to submit issues or pull requests.    
