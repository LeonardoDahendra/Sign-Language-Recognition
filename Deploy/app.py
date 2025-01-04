import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
import gradio as gr

language = 'id'

mp_hands = mp.solutions.hands
mp_faces = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, hand_model, face_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image.flags.writeable = False
    hand_results = hand_model.process(image)
    face_results = face_model.process(image)
    # image.flags.writeable = True
    return image, hand_results, face_results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness[index].classification[0].index == 0:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            ) 
            else:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
                
def extract_keypoints(hand_results, face_results, w, h):
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    face_hand_dif = [1000, 1000, 1000]
    face_size = np.ones(2)
    found = False
    if hand_results.multi_hand_landmarks:
        found = True

        for index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            ref = hand_landmarks.landmark[0]
            ref = np.array([ref.x, ref.y, ref.z])
            min_pos = ref
            max_pos = ref
            all_pos = []
            hand_center = [0, 0]
            for res in hand_landmarks.landmark:
                hand_center = [curr + new for curr, new in zip(hand_center, [res.x, res.y])]
                norm_pos = np.array([res.x - ref[0], res.y - ref[1], res.z])
                all_pos.append(norm_pos)
                min_pos = np.minimum(min_pos, np.array([res.x, res.y, res.z]))
                max_pos = np.maximum(max_pos, np.array([res.x, res.y, res.z]))
            hand_center = [pos / len(hand_landmarks.landmark) for pos in hand_center]
            max_face_pos = [0, 0]
            min_face_pos = [1, 1]
            if face_results.detections:
                for detection in face_results.detections:
                    face_center = [0, 0]
                    for keypoint in detection.location_data.relative_keypoints:
                        face_center = [pos + point for pos, point in zip(face_center, [keypoint.x, keypoint.y])]
                        min_face_pos = np.minimum(min_face_pos, np.array([keypoint.x, keypoint.y]))
                        max_face_pos = np.maximum(max_face_pos, np.array([keypoint.x, keypoint.y]))
                    face_center = [pos / len(detection.location_data.relative_keypoints) for pos in face_center]
                    dif = [face - hand for face, hand in zip(face_center, hand_center)]
                    dif.append(1000)
                    if sum([abs(val) for val in dif]) < sum([abs(val) for val in face_hand_dif]):
                        face_hand_dif = dif
            face_size = np.array(max_face_pos) - np.array(min_face_pos)
            size = max_pos - min_pos
            hand_size = (abs((all_pos[1][0] - all_pos[2][0]) * w) + abs((all_pos[1][1] - all_pos[2][1]) * h) + abs((all_pos[1][2] - all_pos[2][2]) * 2000)) / 10
            approx_z = face_size[0] * face_size[1] * 100 - hand_size
            face_hand_dif[2] = approx_z
            if hand_results.multi_handedness[index].classification[0].index == 0:
                lh = [pos / size for pos in
                            all_pos]
            else:
                rh = [pos / size for pos in
                            all_pos]
    face_hand_dif = np.array(face_hand_dif)
    face_size = np.append(face_size, [1])
    face_hand_dif = face_hand_dif / face_size
    face_hand_dif = face_hand_dif.reshape(1, face_hand_dif.shape[0])
    return np.concatenate([lh, rh, face_hand_dif], axis=0), found

def draw_squares(image, results, name, acc):
    min_x = 1000
    min_y = 1000
    max_x = 0
    max_y = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                min_x = min(landmark.x, min_x)
                min_y = min(landmark.y, min_y)
                max_x = max(landmark.x, max_x)
                max_y = max(landmark.y, max_y)
    min_x = round(min_x * image.shape[1])
    min_y = round(min_y * image.shape[0])
    max_x = round(max_x * image.shape[1])
    max_y = round(max_y * image.shape[0])
    min_x -= 20
    min_y -= 20
    max_x += 20
    max_y += 20
    thickness = 3
    cv2.rectangle(image, (min_x - thickness + 1, min_y - 30), (max_x + thickness - 1, min_y), (0, 255, 0), -1)
    cv2.rectangle(image,(min_x, min_y),(max_x, max_y), (0, 255, 0), thickness)
    cv2.putText(image, name + ": " + str(round(acc, 2)), (min_x + 8, min_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

actions = np.array(['Anda', 'Apa', 'Berhenti', 'Bodoh', 'Cantik', 'Halo', 'Hati-hati', 'Lelah', 'Maaf', 'Makan', 'Mau', 'Membaca', 'Nama', 'Sama-sama', 'Saya', 'Siapa', 'Sombong', 'Takut', 'Terima Kasih'])
def loadModel():
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(43, 3)),
        MaxPool1D(),
        Conv1D(64, 3, activation='relu'),
        MaxPool1D(),
        Conv1D(128, 3, activation='relu'),
        MaxPool1D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(actions.shape[0], activation='softmax')
    ])
    model.load_weights("CNN V3.h5")

    return model
    
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
faces = mp_faces.FaceDetection(min_detection_confidence=0.5)

def main(frame):
    image, hand_results, face_results = mediapipe_detection(frame, hands, faces)
    draw_styled_landmarks(image, hand_results)
    keypoints, visible = extract_keypoints(hand_results, face_results, image.shape[1], image.shape[0])
    if visible:
        pred = model.predict(np.expand_dims(keypoints, axis=0))[0]
        res = np.argmax(pred)

        draw_squares(image, hand_results, actions[res], pred[res])

    return image

model = loadModel()

reference_path = [
    "./Referensi Gerakan/Anda.jpg",
    "./Referensi Gerakan/Apa.jpg",
    "./Referensi Gerakan/Berhenti.jpg",
    "./Referensi Gerakan/Bodoh.jpg",
    "./Referensi Gerakan/Cantik.jpg",
    "./Referensi Gerakan/Halo.jpg",
    "./Referensi Gerakan/Hati-hati.jpg",
    "./Referensi Gerakan/Lelah.jpg",
    "./Referensi Gerakan/Maaf.jpg",
    "./Referensi Gerakan/Makan.jpg",
    "./Referensi Gerakan/Mau Ingin.jpg",
    "./Referensi Gerakan/Membaca.jpg",
    "./Referensi Gerakan/Nama.jpg",
    "./Referensi Gerakan/Sama-sama.jpg",
    "./Referensi Gerakan/Saya.jpg",
    "./Referensi Gerakan/Siapa.jpg",
    "./Referensi Gerakan/Sombong.jpg",
    "./Referensi Gerakan/Takut.jpg",
    "./Referensi Gerakan/Terima Kasih.jpg"
]

interface = gr.Interface(
    fn=main,
    inputs= [
        gr.Image(sources=["webcam"], streaming=True, label="Input Camera"),
    ],
    outputs=[
        gr.Image(label="Output"),
    ],
    live = True
)

if __name__ == "__main__":
    interface.launch(debug=True)
