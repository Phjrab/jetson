import cv2
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 한 손만 집중적으로 인식하도록 설정 (정확도 향상)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 손가락 끝 마디 번호: 검지(8), 중지(12), 약지(16), 새끼(20)
    tip_ids = [8, 12, 16, 20]
    
    while True:
        success, frame = cap.read()
        if not success: break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. 마디 선 그리기
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 2. 손가락 개수 세기 로직
                landmarks = hand_landmarks.landmark
                fingers = []
                
                # 엄지 손가락 (옆으로 펴졌는지 x좌표로 확인)
                # 오른손 기준: 끝(4번)이 마디(3번)보다 왼쪽에 있으면 펴진 것
                if landmarks[4].x < landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 나머지 4개 손가락 (위로 펴졌는지 y좌표로 확인)
                # 끝(tip)이 중간마디(tip-2)보다 위에(y값이 작으면) 있으면 펴진 것
                for tip_id in tip_ids:
                    if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # 펴진 손가락 총 개수
                total_fingers = fingers.count(1)
                
                # 3. 화면에 숫자 크게 표시
                cv2.putText(image, f'Fingers: {total_fingers}', (40, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Finger Counter System</h1><img src='/video_feed' width='640'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)