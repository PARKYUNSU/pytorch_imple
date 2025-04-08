from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import io
import cv2
import numpy as np
import os
import tempfile
from flask_cors import CORS

from detect import detect

app = Flask(__name__)
CORS(app)
# index.html 템플릿 렌더링 (웹 인터페이스 제공)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Video file is missing'}), 400

    video_file = request.files['video']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
        video_file.save(tmp_in)
        input_video_path = tmp_in.name

    try:
        min_score = float(request.form.get('min_score', 0.2))
        max_overlap = float(request.form.get('max_overlap', 0.5))
        top_k = int(request.form.get('top_k', 200))

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Unable to open video file'}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix='.mp4')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            annotated_frame_img = detect(pil_frame, min_score, max_overlap, top_k)
            annotated_frame = cv2.cvtColor(np.array(annotated_frame_img), cv2.COLOR_RGB2BGR)
            out.write(annotated_frame)

        cap.release()
        out.release()

        response = send_file(
            output_video_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='annotated_video.mp4'
        )

        response.call_on_close(lambda: os.remove(output_video_path))
        return response

    finally:
        os.remove(input_video_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)