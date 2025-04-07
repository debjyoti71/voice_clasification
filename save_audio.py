import serial
import wave
import struct

SERIAL_PORT = "COM12"
BAUD_RATE = 115200
DURATION_SECONDS = 5
SAMPLE_RATE = 16000

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
output_file = "filtered_audio.wav"

wav_file = wave.open(output_file, "wb")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(SAMPLE_RATE)

print("Recording audio with noise filtering...")

samples_received = 0
max_samples = DURATION_SECONDS * SAMPLE_RATE

while samples_received < max_samples:
    data = ser.read(2)
    if len(data) == 2:
        sample = struct.unpack('<h', data)[0]  # Convert bytes to integer (16-bit)
        if abs(sample) > 500:  # Ignore small noise values
            wav_file.writeframes(data)
            samples_received += 1

print(f"Recording complete. Saved as {output_file}")

wav_file.close()
ser.close()
