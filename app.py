from flask import Flask, request
import wave

app = Flask(__name__)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_data = request.data

    with wave.open("recorded_audio.wav", "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    return "Audio received", 200

if __name__ == '__main__':
    app.run()
