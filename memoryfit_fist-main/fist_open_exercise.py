import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def test():
    # MediaPipe 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    # 모델 로드
    model = tf.keras.models.load_model('fist_open_wrong_model.h5')

    # 라벨 인코더 로드 및 적합
    label_encoder = LabelEncoder()
    label_encoder.fit(['fist', 'open', 'wrong'])

    # 웹캠을 통해 영상을 캡처합니다.
    cap = cv2.VideoCapture(0)

    # 초기 상태와 카운트 변수
    previous_prediction = 'unknown'
    count = 0

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
                    landmarks = [[lm.x, lm.y, lm.z]
                                for lm in hand_landmarks.landmark]
                    landmarks_flat = np.array(landmarks).flatten().tolist()

                    # 예측을 위해 데이터 형태 변환
                    landmarks_array = np.array([landmarks_flat])
                    landmarks_array = landmarks_array.reshape(
                        (landmarks_array.shape[0], 1, landmarks_array.shape[1]))

                    # 예측
                    probabilities = model.predict(landmarks_array)[0]
                    predicted_label_index = np.argmax(probabilities)
                    predicted_label = label_encoder.inverse_transform(
                        [predicted_label_index])[0]
                    confidence = probabilities[predicted_label_index]

                    predictions.append(predicted_label)
                    confidences.append(confidence)

                # 모든 예측이 동일한지 확인하고, 가장 높은 확률을 가진 예측을 선택
                if all(pred == predictions[0] for pred in predictions):
                    final_prediction = predictions[np.argmax(confidences)]
                else:
                    final_prediction = 'wrong'

                # 이전 예측이 fist이고 현재 예측이 open이면 count를 증가시킴
                if previous_prediction == 'fist' and final_prediction == 'open':
                    count += 1
                    print(f"Count: {count}")

                previous_prediction = final_prediction

                print(f"Prediction: {final_prediction}")

            else:
                final_prediction = "unknown"
                print(final_prediction)

            # 이미지 좌우 반전
            flipped_image = cv2.flip(image, 1)

            # 반전된 이미지에 텍스트 추가
            cv2.putText(flipped_image, f"Prediction: {final_prediction}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(flipped_image, f"Count: {count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 카운트가 30에 도달하면 중앙에 "목표 달성" 텍스트 표시
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

            # 반전된 이미지 보여주기
            cv2.imshow('MediaPipe Hands', flipped_image)

            # ESC 키를 누르면 종료합니다.
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
