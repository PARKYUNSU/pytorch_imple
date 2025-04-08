// server.js
const express = require('express');
const axios = require('axios');
const multer = require('multer'); // 파일 업로드를 위한 미들웨어
const path = require('path');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' }); // 임시 폴더에 업로드 파일 저장

// 정적 파일 (index.html 등)을 제공할 public 폴더 설정
app.use(express.static(path.join(__dirname, 'public')));

// 기본 URL 접속 시 index.html 제공
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 클라이언트의 요청을 받아 Flask의 /detect_video API로 요청을 전달
app.post('/api/detect_video', upload.single('video'), async (req, res) => {
    try {
        // req.file에는 업로드된 파일 정보가 들어있습니다.
        const videoPath = req.file.path;

        // FormData를 구성하여 파일과 옵션들을 Flask로 전송합니다.
        const FormData = require('form-data');
        const formData = new FormData();
        formData.append('video', fs.createReadStream(videoPath));
        formData.append('min_score', '0.5');
        formData.append('max_overlap', '0.5');
        formData.append('top_k', '200');

        // Flask 서버의 엔드포인트 URL (예: http://127.0.0.1:5001/detect_video)
        const flaskURL = 'http://127.0.0.1:5001/detect_video';

        // axios를 사용해 POST 요청 (헤더에 formData.getHeaders() 필수)
        const response = await axios.post(flaskURL, formData, {
            headers: formData.getHeaders(),
            responseType: 'stream', // 영상 파일을 스트림 형태로 처리
        });

        // Flask API가 반환한 파일 스트림을 그대로 Node.js 클라이언트에 전달
        res.setHeader('Content-Type', response.headers['content-type']);
        res.setHeader('Content-Disposition', response.headers['content-disposition']);
        response.data.pipe(res);

        // 파일 사용 후, 로컬에 저장된 업로드 파일 삭제
        fs.unlink(videoPath, (err) => {
            if (err) console.error('Failed to delete uploaded file: ', err);
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal server error: ' + error.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Node.js server is running on port ${PORT}`);
});
