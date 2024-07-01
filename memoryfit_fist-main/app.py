from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

model = tf.keras.models.load_model('fist_open_wrong_model.h5')
label_encoder = LabelEncoder()
label_encoder.fit(['fist', 'open', 'wrong'])

previous_prediction = 'unknown'
count = 0


def generate_frames():
    global previous_prediction, count

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                predictions = []
                confidences = []

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    landmarks = [[lm.x, lm.y, lm.z]
                                 for lm in hand_landmarks.landmark]
                    landmarks_flat = np.array(landmarks).flatten().tolist()

                    landmarks_array = np.array([landmarks_flat])
                    landmarks_array = landmarks_array.reshape(
                        (landmarks_array.shape[0], 1, landmarks_array.shape[1]))

                    probabilities = model.predict(landmarks_array)[0]
                    predicted_label_index = np.argmax(probabilities)
                    predicted_label = label_encoder.inverse_transform(
                        [predicted_label_index])[0]
                    confidence = probabilities[predicted_label_index]

                    predictions.append(predicted_label)
                    confidences.append(confidence)

                if all(pred == predictions[0] for pred in predictions):
                    final_prediction = predictions[np.argmax(confidences)]
                else:
                    final_prediction = 'wrong'

                if previous_prediction == 'fist' and final_prediction == 'open':
                    count += 1
                    print(f"Count: {count}")

                previous_prediction = final_prediction

                print(f"Prediction: {final_prediction}")

            else:
                final_prediction = "unknown"
                print(final_prediction)

            flipped_image = cv2.flip(image, 1)

            cv2.putText(flipped_image, f"Prediction: {final_prediction}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(flipped_image, f"Count: {count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if count >= 30:
                text = "SUCCESS"
                font_scale = 2
                font_thickness = 3
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = (flipped_image.shape[1] - text_size[0]) // 2
                text_y = (flipped_image.shape[0] + text_size[1]) // 2
                cv2.putText(flipped_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', flipped_image)
            flipped_image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + flipped_image + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
