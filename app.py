from flask import Flask, request
import wave
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is running on Render!"

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_data = request.data
    audio_file_path = "recorded_audio.wav"

    with wave.open(audio_file_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    return "Audio received", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render requires PORT env
    app.run(host='0.0.0.0', port=port)
