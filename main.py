import cv2 as cv
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

webcam = cv.VideoCapture(0)
success, frame = webcam.read()
while success:
    frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)

    # To improve performance
    frame.flags.writeable = False

    # Get the result
    results = face_mesh.process(frame)

    # To improve performance
    frame.flags.writeable = True

    # Convert the colour space from RGB to BGR
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    frame_height, frame_width, frame_channels = frame.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index, landmark in enumerate(face_landmarks.landmark):
                if index == 33 or index == 263 or index == 1 or index == 61 or index == 291 or index == 199:
                    if index == 1:
                        nose_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        nose_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)

                    # Get the 2d coordinates
                    face_2d.append([x, y])

                    # Get the 3d coordinates
                    face_3d.append([x, y, landmark.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * frame_width
            camera_matrix = np.array([[focal_length, 0, frame_height / 2],
                                      [0, focal_length, frame_width / 2],
                                      [0, 0, 1]])

            # The distortion parameters
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # rot_vec is the rotation vector
            # trans_vec is the translation vector
            success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d,
                                                      camera_matrix,
                                                      distortion_matrix)

            # Get rotational matrix
            rotation_matrix, jacobian_matrix = cv.Rodrigues(rot_vec)

            # mtxR:  The matrix R, an upper triangular matrix representing
            # the intrinsic camera parameters when decomposing a camera matrix
            # mtxQ:  The matrix Q, an orthogonal matrix resulting from the
            # decomposition, representing rotation or orientation in 3D space.
            # Qx:  Rotation matrix around the x-axis
            # Qy:  Rotation matrix around the y-axis
            # Qz:  Rotation matrix around the z-axis
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            # Convert `angles` to the degree system
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Find the direction at which the user's head is directed
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

            # Display the normal vector at the nose
            nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec,
                                                            trans_vec,
                                                            camera_matrix,
                                                            distortion_matrix)
            point1 = (int(nose_2d[0]), int(nose_2d[1]))
            point2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv.line(frame, point1, point2, (255, 0, 0), 3)

            # Adding text to the frame
            cv.putText(frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv.putText(frame, f"x: {np.round(x, 2)}", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, f"y: {np.round(y, 2)}", (500, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, f"z: {np.round(z, 2)}", (500, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)
            cv.imshow('Webcam', frame)

            if cv.waitKey(5) & 0xFF == 27:
                break

    success, frame = webcam.read()

webcam.release()
