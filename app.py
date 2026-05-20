from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os
import base64
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024 # 5 GB video upload limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

model = YOLO('models/best_v3.pt')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def filter_huge_boxes(results, max_area_ratio=0.85):
    """Filtre: Görüntünün %85'inden fazlasını kaplayan devasa hatalı bounding box'ları eler. Yakın çekimlere izin verir."""
    if len(results[0].boxes) == 0:
        return results
        
    image_area = results[0].orig_shape[0] * results[0].orig_shape[1]
    valid_indices = []
    
    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes[i].xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= image_area * max_area_ratio:
            valid_indices.append(i)
            
    if len(valid_indices) < len(results[0].boxes):
        results[0] = results[0][valid_indices]
            
    return results

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    results = model(img_path, conf=0.35)
    results = filter_huge_boxes(results)
    result_img = results[0].plot()
    result_path = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
    Image.fromarray(result_img[..., ::-1]).save(result_path)

    detections = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append({'class': cls_name, 'confidence': round(conf * 100, 1)})

    with open(result_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({'detections': detections, 'result_image': img_base64, 'total': len(detections), 'type': 'image'})

@app.route('/detect_video', methods=['POST'])
def detect_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya yüklenmedi'}), 400
        file = request.files['file']

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.close() # Windows'ta dosya kilitleme hatasını (WinError 32) önlemek için kapatıyoruz
        file.save(tmp.name)

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return jsonify({'error': 'Video okunamadı. Desteklenmeyen format veya bozuk dosya olabilir.'}), 400
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        out_path = os.path.join(UPLOAD_FOLDER, 'result_video.webm')
        # Tarayıcıda en kusursuz ve codec bağımsız çalışan format WebM (VP8) formatıdır.
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
 
        all_detections = {}
        frame_count = 0
        sample_frame_b64 = None
        seen_tracks = set()
        track_history = {}
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLOv8 nesne takip modunu (BoT-SORT) kullanarak her yabancı ota benzersiz ID veriyoruz
            results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.35, verbose=False)
            results = filter_huge_boxes(results)
            annotated = results[0].plot()
            
            if results[0].boxes.id is not None:
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    track_id = int(box.id[0])
                    
                    # Otun gerçekten var olduğundan emin olmak için 3 kare (frame) boyunca stabil kalmasını bekle
                    track_history[track_id] = track_history.get(track_id, 0) + 1
                    
                    if track_history[track_id] == 3:
                        unique_key = f"{cls_name}_{track_id}"
                        if unique_key not in seen_tracks:
                            seen_tracks.add(unique_key)
                            if cls_name not in all_detections:
                                all_detections[cls_name] = []
                            all_detections[cls_name].append(round(conf * 100, 1))
            else:
                # Takip ID'si henüz atanmadıysa (ilk kareler), o karedeki maksimum eşzamanlı ot sayısını baz alarak tekilleştirme yap
                current_frame_counts = {}
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])]
                    current_frame_counts[cls_name] = current_frame_counts.get(cls_name, 0) + 1
                
                for cls_name, count in current_frame_counts.items():
                    if cls_name not in all_detections:
                        all_detections[cls_name] = []
                    while len(all_detections[cls_name]) < count:
                        all_detections[cls_name].append(80.0)
            
            if sample_frame_b64 is None and len(results[0].boxes) > 0:
                _, buf = cv2.imencode('.jpg', annotated)
                sample_frame_b64 = base64.b64encode(buf).decode('utf-8')
            out.write(annotated)
            frame_count += 1
 
        cap.release()
        out.release()
        os.unlink(tmp.name)

        if sample_frame_b64 is None:
            cap2 = cv2.VideoCapture(out_path)
            ret, frame = cap2.read()
            if ret:
                _, buf = cv2.imencode('.jpg', frame)
                sample_frame_b64 = base64.b64encode(buf).decode('utf-8')
            cap2.release()

        summary = [{'class': k, 'count': len(v), 'avg_confidence': round(sum(v)/len(v), 1)} for k, v in all_detections.items()]

        return jsonify({'detections': summary, 'total': sum(d['count'] for d in summary), 'frames': frame_count, 'sample_frame': sample_frame_b64, 'type': 'video', 'video_url': '/static/uploads/result_video.webm'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Sistemsel hata: {str(e)}'}), 500

@socketio.on('process_frame')
def handle_process_frame(data):
    try:
        # PWA veya web üzerinden gelen veriyi al
        if isinstance(data, dict):
            img_data = data.get('image').split(',')[1]
        else:
            img_data = data.split(',')[1]

        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(frame, conf=0.35, verbose=False)
        results = filter_huge_boxes(results)
        annotated = results[0].plot()
        
        detections = []
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({'class': cls_name, 'confidence': round(conf * 100, 1)})

        if len(detections) > 0:
            _, buffer = cv2.imencode('.jpg', annotated)
            result_img_base64 = base64.b64encode(buffer).decode('utf-8')
            emit('frame_result', {
                'image': 'data:image/jpeg;base64,' + result_img_base64,
                'detections': detections
            })
    except Exception as e:
        print("Frame error:", e)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
