import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json

# 모델 로드
model = tf.keras.models.load_model('pandas/apart_touch_wrong_model.h5')

# 라벨 인코더 로드 및 적합
label_encoder = LabelEncoder()
label_encoder.fit(['apart', 'touch', 'wrong'])

# 카운트 변수 초기화
count = 0
touch_detected = False

# MediaPipe 초기화
mp_hands = mp.solutions.hands

def process_frame():
    global count, touch_detected

    # 웹캠을 통해 영상을 캡처합니다.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: 카메라를 찾을 수 없습니다.")
        return

    print("카메라가 성공적으로 열렸습니다.")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error: 프레임을 읽을 수 없습니다.")
                break

            # 성능 향상을 위해 이미지 작성 가능성을 비활성화합니다.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 이미지 작성 가능성을 다시 활성화합니다.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                predictions = []
                confidences = []

                for hand_landmarks in results.multi_hand_landmarks:
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
                if final_prediction == 'touch':
                    touch_detected = True
                elif final_prediction == 'apart' and touch_detected:
                    count += 1
                    touch_detected = False  # Reset touch_detected after counting
                elif final_prediction == 'wrong':
                    # If wrong is detected, maintain the touch_detected state but do not increase count
                    touch_detected = touch_detected

                print(f"Prediction: {final_prediction}, Count: {count}")

            else:
                final_prediction = "unknown"
                print(final_prediction)

            # ESC 키를 누르면 종료합니다.
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수로 실행될 때만 작동하게 함
if __name__ == "__main__":
    process_frame()
    result = {
        'count': count,
        'message': 'Prediction process completed'
    }
    print(json.dumps(result))
