import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
import signal  # 추가: 종료 신호 감지
import sys     # 추가: 시스템 종료

app = Flask(__name__)

# 전역 변수 설정
current_count = 0
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 카메라를 전역으로 미리 열어둠

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- [추가] 자동 정리 함수 ---
def cleanup_resources(sig, frame):
    print('\n[시스템] 종료 신호를 감지했습니다. 자원을 해제합니다...')
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print('[시스템] 카메라가 안전하게 끄졌습니다. 퇴근!')
    sys.exit(0)

# Ctrl+C (SIGINT) 신호가 들어오면 cleanup_resources 함수를 실행하도록 등록
signal.signal(signal.SIGINT, cleanup_resources)
# ---------------------------

def generate_frames():
    global current_count
    # 이제 cap은 밖에서 전역으로 관리하므로 여기서 새로 열지 않습니다.
    tip_ids = [8, 12, 16, 20]
    
    while True:
        success, frame = cap.read()
        if not success: break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                fingers = []
                
                # 엄지 계산
                if landmarks[4].x < landmarks[3].x: fingers.append(1)
                else: fingers.append(0)
                
                # 나머지 손가락 계산
                for tip_id in tip_ids:
                    if landmarks[tip_id].y < landmarks[tip_id - 2].y: fingers.append(1)
                    else: fingers.append(0)
                count = fingers.count(1)
        
        current_count = count
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    return jsonify(count=current_count)

@app.route('/')
def index():
    # (기존 HTML 코드는 동일하므로 유지됩니다)
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson AI Dashboard</title>
        <style>
            body { background-color: #121212; color: #e0e0e0; font-family: sans-serif; margin: 0; padding: 20px; }
            .header { display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px; }
            .main-layout { display: flex; gap: 20px; }
            .video-container { flex: 2; background: #000; border-radius: 12px; border: 1px solid #444; }
            .video-container img { width: 100%; display: block; }
            .side-panel { flex: 1; display: flex; flex-direction: column; gap: 15px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; border: 1px solid #333; text-align: center; }
            .card h3 { margin: 0 0 10px 0; color: #00e676; font-size: 0.9rem; }
            .count-display { font-size: 5rem; font-weight: bold; color: #fff; margin: 10px 0; }
            .status-btn { background: #2e7d32; color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; }
        </style>
        <script>
            setInterval(function() {
                fetch('/get_count')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('finger-count').innerText = data.count;
                    });
            }, 100);
        </script>
    </head>
    <body>
        <div class="header"><h2>Autonomous Design Center</h2></div>
        <div class="main-layout">
            <div class="video-container"><img src="/video_feed"></div>
            <div class="side-panel">
                <div class="card">
                    <h3>FINGER COUNT</h3>
                    <div id="finger-count" class="count-display">0</div>
                    <p>Real-time Tracking</p>
                </div>
                <div class="card">
                    <h3>SYSTEM STATUS</h3>
                    <p>Jetson Orin Nano: <span style="color:#00e676">Online</span></p>
                    <button class="status-btn">CONTROL ACTIVE</button>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    # threaded=True를 사용하여 스트리밍과 API 응답을 동시에 처리
    app.run(host='0.0.0.0', port=5000, threaded=True)