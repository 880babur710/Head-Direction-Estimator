import cv2 as cv
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_face_landmarks(frame):
    """Process the frame to get face landmarks."""
    frame_rgb = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
    # To improve performance
    frame_rgb.flags.writeable = False
    # Get the result
    results = face_mesh.process(frame_rgb)
    # To improve performance
    frame_rgb.flags.writeable = True
    # Convert the colour space from RGB to BGR
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

    return frame_bgr, results


def get_facial_features(frame, face_landmarks):
    """Extract 2D and 3D facial features."""
    frame_height, frame_width, _ = frame.shape
    face_2d = []
    face_3d = []
    nose_2d = None
    nose_3d = None

    landmark_indices = [33, 263, 1, 61, 291, 199]

    for index in landmark_indices:
        landmark = face_landmarks.landmark[index]
        x, y = landmark.x * frame_width, landmark.y * frame_height

        if index == 1:
            nose_2d = (x, y)
            nose_3d = (x, y, landmark.z * 3000)

        # Get the 2d coordinates
        face_2d.append([int(x), int(y)])
        # Get the 3d coordinates
        face_3d.append([int(x), int(y), landmark.z])

    face_2d_array = np.array(face_2d, dtype=np.float64)
    face_3d_array = np.array(face_3d, dtype=np.float64)

    return face_2d_array, face_3d_array, nose_2d, nose_3d


def calculate_head_pose(face_2d, face_3d, frame):
    """Calculate the head pose using solvePnP."""
    frame_height, frame_width, _ = frame.shape
    focal_length = 1 * frame_width
    camera_matrix = np.array([[focal_length, 0, frame_height / 2],
                              [0, focal_length, frame_width / 2],
                              [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, camera_matrix,
                                              distortion_matrix)
    if not success:
        return None, None, None

    rotation_matrix, _ = cv.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv.RQDecomp3x3(rotation_matrix)
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    return x, y, z, rot_vec, trans_vec, camera_matrix, distortion_matrix


def display_results(frame, x, y, z, nose_2d, nose_3d, rot_vec, trans_vec, camera_matrix, distortion_matrix):
    """Display the results on the frame."""
    if y > 10:
        text = "Looking Right"
    elif y < -10:
        text = "Looking Left"
    elif x > 10:
        text = "Looking Up"
    elif x < -10:
        text = "Looking Down"
    else:
        text = "Looking Forward"

    nose_3d_projection, _ = cv.projectPoints(nose_3d, rot_vec, trans_vec,
                                             camera_matrix, distortion_matrix)
    point1 = (int(nose_2d[0]), int(nose_2d[1]))
    point2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    cv.line(frame, point1, point2, (255, 0, 0), 3)
    cv.putText(frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv.putText(frame, f"x: {np.round(x, 2)}", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(frame, f"y: {np.round(y, 2)}", (500, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(frame, f"z: {np.round(z, 2)}", (500, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def main():
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        success, frame = webcam.read()
        while success:
            frame, results = get_face_landmarks(frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_2d, face_3d, nose_2d, nose_3d = get_facial_features(frame, face_landmarks)
                    x, y, z, rot_vec, trans_vec, camera_matrix, distortion_matrix = calculate_head_pose(face_2d, face_3d, frame)
                    if x is not None and y is not None and z is not None:
                        display_results(frame, x, y, z, nose_2d, nose_3d, rot_vec, trans_vec, camera_matrix, distortion_matrix)
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)

            cv.imshow('Webcam', frame)
            if cv.waitKey(5) & 0xFF == 27:
                break

            success, frame = webcam.read()
    finally:
        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
