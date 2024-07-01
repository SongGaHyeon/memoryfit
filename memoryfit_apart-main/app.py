from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import threading

app = Flask(__name__)

# 모델 로드
model = tf.keras.models.load_model('pandas/apart_touch_wrong_model.h5')

# 라벨 인코더 로드 및 적합
label_encoder = LabelEncoder()
label_encoder.fit(['apart', 'touch', 'wrong'])

# 카운트 변수 초기화
count = 0
touch_detected = False
lock = threading.Lock()

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

def generate_frames(camera_index=0):
    global count, touch_detected
    cap = cv2.VideoCapture(camera_index)  # 카메라 인덱스를 지정

    if not cap.isOpened():
        print("Error: 카메라를 찾을 수 없습니다.")
        return

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 성능 향상을 위해 이미지 작성 가능성을 비활성화합니다.
            frame.flags.writeable = False
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 이미지 작성 가능성을 다시 활성화합니다.
            frame.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 올바른 색상 공간으로 변환

            final_prediction = "unknown"  # 기본값으로 초기화
            confidence = 0.0  # 기본값으로 초기화

            if results.multi_hand_landmarks:
                predictions = []
                confidences = []

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 랜드마크 데이터 추출
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    landmarks_flat = np.array(landmarks).flatten().tolist()

                    # 예측을 위해 데이터 형태 변환
                    landmarks_array = np.array([landmarks_flat])
                    landmarks_array = landmarks_array.reshape((landmarks_array.shape[0], 1, landmarks_array.shape[1]))

                    # 예측
                    probabilities = model.predict(landmarks_array)[0]
                    predicted_label_index = np.argmax(probabilities)
                    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
                    confidence = probabilities[predicted_label_index]

                    predictions.append(predicted_label)
                    confidences.append(confidence)

                # 모든 예측이 동일한지 확인하고, 가장 높은 확률을 가진 예측을 선택
                if all(pred == predictions[0] for pred in predictions):
                    final_prediction = predictions[np.argmax(confidences)]
                else:
                    final_prediction = 'wrong'

                # touch와 apart 한쌍 감지 로직
                with lock:
                    if final_prediction == 'touch':
                        touch_detected = True
                    elif final_prediction == 'apart' and touch_detected:
                        count += 1
                        touch_detected = False  # Reset touch_detected after counting
                    elif final_prediction == 'wrong':
                        touch_detected = touch_detected

            # 프레임 반전
            flipped_frame = cv2.flip(frame, 1)

            # 결과를 프레임에 추가
            with lock:
                cv2.putText(flipped_frame, f"Prediction: {final_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(flipped_frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(flipped_frame, f"Count: {count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # count가 30이 되면 success 표시
                if count >= 30:
                    text = "SUCCESS"
                    font_scale = 2
                    font_thickness = 3
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = (flipped_frame.shape[1] - text_size[0]) // 2
                    text_y = (flipped_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(flipped_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', flipped_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(camera_index=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count')
def get_count():
    with lock:
        return jsonify({'count': count})

if __name__ == '__main__':
    app.run(debug=True)
