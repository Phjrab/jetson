import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# 전역 변수: 얼굴 상태 저장
face_status = "Scanning..."

# MediaPipe Face Mesh 초기화 (홍채 인식 포함)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# refine_landmarks=True 설정으로 눈동자(홍채)까지 정밀 추적
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def generate_frames():
    global face_status
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
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
                
                # 2. 입 벌림 감지 로직 (윗입술 13번, 아랫입술 14번 점 거리)
                upper_lip = face_landmarks.landmark[13].y
                lower_lip = face_landmarks.landmark[14].y
                mouth_distance = lower_lip - upper_lip
                
                if mouth_distance > 0.05: # 기준값은 환경에 따라 조절 가능
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