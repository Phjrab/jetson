import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
import signal
import sys

app = Flask(__name__)

# 전역 변수 설정
face_status = "Scanning..."
# [수정] 카메라를 전역에서 한 번만 열도록 설정
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- [추가] 자동 정리 함수 ---
def cleanup_resources(sig, frame):
    print('\n[시스템] 종료 신호를 감지했습니다. Face Mesh 엔진을 정지합니다...')
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print('[시스템] 카메라 자원 해제 완료. 안전하게 종료되었습니다.')
    sys.exit(0)

# Ctrl+C 신호 등록
signal.signal(signal.SIGINT, cleanup_resources)
# ---------------------------

def generate_frames():
    global face_status
    # 이제 cap을 함수 내부에서 새로 생성하지 않고 전역 변수를 사용합니다.
    
    while True:
        success, frame = cap.read()
        if not success: break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        status = "No Face Detected"
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. 얼굴 전체 망(Mesh) 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                
                # 2. 입 벌림 감지 로직
                upper_lip = face_landmarks.landmark[13].y
                lower_lip = face_landmarks.landmark[14].y
                mouth_distance = lower_lip - upper_lip
                
                if mouth_distance > 0.05:
                    status = "Mouth Open"
                else:
                    status = "Face Active"

        face_status = status
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify(status=face_status)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson Face Perception</title>
        <style>
            body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
            .container { display: flex; gap: 20px; max-width: 1200px; margin: auto; }
            .video-box { flex: 2; background: #1e293b; padding: 10px; border-radius: 20px; border: 2px solid #334155; }
            .info-box { flex: 1; display: flex; flex-direction: column; gap: 20px; }
            .card { background: #1e293b; padding: 25px; border-radius: 20px; border: 1px solid #334155; text-align: center; }
            h2 { color: #38bdf8; margin-top: 0; font-size: 1.2rem; }
            .status-text { font-size: 2rem; font-weight: bold; color: #facc15; }
            img { width: 100%; border-radius: 15px; }
        </style>
        <script>
            setInterval(function() {
                fetch('/get_status').then(res => res.json()).then(data => {
                    document.getElementById('status').innerText = data.status;
                });
            }, 200);
        </script>
    </head>
    <body>
        <div style="text-align:center; margin-bottom:30px;">
            <h1>Face Analysis Dashboard</h1>
            <p>Powered by Jetson Orin Nano & MediaPipe</p>
        </div>
        <div class="container">
            <div class="video-box"><img src="/video_feed"></div>
            <div class="info-box">
                <div class="card">
                    <h2>SYSTEM STATUS</h2>
                    <div id="status" class="status-text">Scanning...</div>
                </div>
                <div class="card">
                    <h2>ENGINE</h2>
                    <p>Mediapipe Face Mesh</p>
                    <p style="color:#94a3b8">468 Landmarks Active</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)