from flask import Flask, request, send_file
import os

app = Flask(__name__)

AUDIO_FILE = "latest_audio.wav"

@app.route('/upload', methods=['POST'])
def upload_audio():
    with open(AUDIO_FILE, 'wb') as f:
        f.write(request.data)
    return "Audio Saved", 200

@app.route('/latest-audio', methods=['GET'])
def get_audio():
    if os.path.exists(AUDIO_FILE):
        return send_file(AUDIO_FILE, mimetype="audio/wav")
    return "No audio found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Use Render's assigned port
