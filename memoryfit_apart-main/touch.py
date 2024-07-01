import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# 모델 로드
model = tf.keras.models.load_model('pandas/apart_touch_wrong_model.h5')

# 라벨 인코더 로드 및 적합
label_encoder = LabelEncoder()
label_encoder.fit(['apart', 'touch', 'wrong'])

# 카운트 변수 초기화
count = 0
touch_detected = False

# 웹캠을 통해 영상을 캡처합니다.
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
                mp_drawing.draw_landmarks(
                    image,
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
            if final_prediction == 'touch':
                touch_detected = True
            elif final_prediction == 'apart' and touch_detected:
                count += 1
                touch_detected = False  # Reset touch_detected after counting
            elif final_prediction == 'wrong':
                # If wrong is detected, maintain the touch_detected state but do not increase count
                touch_detected = touch_detected

            print(f"Prediction: {final_prediction}, Count: {count}")

            # 이미지에 예측 결과 및 카운트 표시
            cv2.putText(image, f"Prediction: {final_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # count가 30이 되면 success 표시
            if count >= 30:
                cv2.putText(image, f"SUCCESS", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            final_prediction = "unknown"
            print(final_prediction)
            cv2.putText(image, f"Prediction: {final_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # ESC 키를 누르면 종료합니다.
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
