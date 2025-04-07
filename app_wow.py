import os
import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr
import pyaudio
import wave
import parselmouth
import joblib
import json
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import ADASYN
from scipy.spatial.distance import cityblock
from scipy.stats import pearsonr

MODEL_FILE = "voice_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"
VOICE_DATA_DIR = "voice_samples"
FEATURES_CSV = "audio_features_with_errors.csv"
PREDICT_CSV = "predictvoice.csv"
SPLIT_SENTENCES = [
    "the quick brown", "fox jumps over", "the lazy dog",
    "and the quick brown", "fox jumps high", "over the lazy dog", "in the morning"
]


def record_audio(output_file, duration=5, rate=22050, channels=1):
    print("Recording audio...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
    
    frames = [stream.read(1024) for _ in range(int(rate / 1024 * duration))]

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {output_file}")
    return output_file

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio).lower()
        except (sr.UnknownValueError, sr.RequestError):
            return None


def extract_formants(file_path):
    # Load the audio file
    def add_noise(y, noise_level=0.02):
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise

    # Function to add echo effect
    def add_echo(y, sr, delay=0.2, decay=0.5):
        delay_samples = int(sr * delay)
        echo_signal = np.zeros_like(y)
        echo_signal[delay_samples:] = y[:-delay_samples] * decay
        return y + echo_signal

    # Function to boost loudness (increase dB)
    def increase_loudness(y, gain_db=10):
        gain = 10 ** (gain_db / 20)  # Convert dB to linear scale
        return y * gain

    # Function to apply a high-pass filter (emphasize high frequencies)
    def high_pass_filter(y, sr, cutoff=3000):
        return librosa.effects.preemphasis(y, coef=0.97)  # Simple high-pass effect

    # Store extracted features
    data = []

    y, sr = librosa.load(file_path, sr=22050)

    # Different error conditions
    error_conditions = {
    "Original": y,
    "Noise": add_noise(y),
    "Echo": add_echo(y, sr),
    "High dB": increase_loudness(y),
    "High Frequency": high_pass_filter(y, sr)
    }

    for condition, y_mod in error_conditions.items():
        f0, voiced_flag, voiced_probs = librosa.pyin(y_mod, fmin=50, fmax=500)

        features = {
            "Mean Pitch (F0)": np.nanmean(f0) if f0 is not None else None,
            "Zero Crossing Rate": np.mean(librosa.feature.zero_crossing_rate(y_mod)),
            "Spectral Centroid": np.mean(librosa.feature.spectral_centroid(y=y_mod, sr=sr)),
            "Spectral Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y_mod, sr=sr)),
            "Spectral Flatness": np.mean(librosa.feature.spectral_flatness(y=y_mod)),
            "Spectral Roll-off": np.mean(librosa.feature.spectral_rolloff(y=y_mod, sr=sr)),
            "Root Mean Square Energy": np.mean(librosa.feature.rms(y=y_mod)),
            "Skewness": skew(y_mod) if len(y_mod) > 0 else None,
        }

        mfccs = librosa.feature.mfcc(y=y_mod, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"MFCC {i+1}"] = np.mean(mfccs[i])

        # Additional chroma features
        try:
            chroma_stft = librosa.feature.chroma_stft(y=y_mod, sr=sr)
            features["Chroma STFT"] = np.mean(chroma_stft)
        except:
            features["Chroma STFT"] = None

        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=y_mod, sr=sr)
            features["Chroma CQT"] = np.mean(chroma_cqt)
        except:
            features["Chroma CQT"] = None

        try:
            mel_spec = librosa.feature.melspectrogram(y=y_mod, sr=sr)
            features["Mel Spectrogram"] = np.mean(mel_spec)
        except:
            features["Mel Spectrogram"] = None

        label = "unknown"  # Extract label from path
        features["Label"] = label
        features["Condition"] = condition  # Add error condition
        data.append(features)
        print(f"Extracted features for {condition}: {features}")
    
    return data

def train_model(features_csv):
    """Trains a machine learning model using multiple classifiers and saves the best model."""
    tdf = pd.read_csv(features_csv)
    tdf = tdf.drop("Condition", axis=1)
    tdf = tdf.dropna()

    X = tdf.drop("Label", axis=1)
    y = tdf["Label"]

    le = LabelEncoder()
    y = le.fit_transform(y)
    y= pd.Series(y)

    # print("Class Mapping:", label_map)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    modelp = model.predict(X)
    print("Random Forest Accuracy:", accuracy_score(y, modelp))
    print("RandomForestClassifier:", model.score(X, y))

    joblib.dump(model,MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    # joblib.dump(scaling_params,SCALER_FILE)

def predict(feature_csv):
    """Predicts the speaker using the trained model and prints all confidence levels."""
    df = feature_csv
    model = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    
    pdf = df.drop(["Condition", "Label"], axis=1)
    pdf = pdf.dropna()
    # predict_new = pdf
    predict_new = pdf.mean()

    # Convert to DataFrame with proper column name
    predict_new = pd.DataFrame(predict_new).T 
    print(pdf)

    print("\nPrediction\tConfidence")
    print("-" * 30)

    p = model.predict(predict_new)
    confidence_scores = model.predict_proba(predict_new)

    # Convert predicted labels back to original labels
    p = le.inverse_transform(p)

    # Store results
    predictions = []

    # Print results
    for i, label in enumerate(p):
        conf = max(confidence_scores[i])  # Highest probability (confidence)
        print(f"Predicted Label: {label}, Confidence: {conf:.4f}")
        predictions.append((label, conf))  # Store the label and confidence

    return predictions  # Return all predicted labels and their confidence

def new_input():
    """Takes new user input and collects voice samples."""
    name = input("Enter name: ").strip().lower()
    user_dir = os.path.join(VOICE_DATA_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    all_features = []
    
    for i, part in enumerate(SPLIT_SENTENCES):
        while True:  # Loop until correct audio is recorded
            print(f"\nPart {i+1}: {part}")
            file_path = os.path.join(user_dir, f"sample{i+1}.wav")
            output_path = record_audio(file_path)

            transcribed_text = convert_audio_to_text(output_path)
            if transcribed_text.strip().lower() == part.strip().lower():
                break  # Correct sample, move to the next one
            print(f"Error: Mismatch in sample {i+1}, please retry.")

    for file in os.listdir(user_dir):
        file_path = os.path.join(user_dir, file)
        features = extract_formants(file_path)
        all_features.append(features)

    new_features_df = pd.DataFrame(all_features)

    if os.path.exists("audio_features_with_errors.csv"):
        features_df = pd.read_csv("audio_features_with_errors.csv")
        new_features_df = pd.concat([features_df, new_features_df], ignore_index=True)

    new_features_df.to_csv("audio_features_with_errors.csv", index=False)

def predict_voice():
    """Predicts a user's voice from an audio sample."""
    sentence = "open the door"
    file_path = "predictvoice.wav"
    record_audio(file_path)
    feature= extract_formants(file_path)
    transcribed_text = convert_audio_to_text(file_path)
    print(f"You said: {transcribed_text}")

    if feature:
        feature_df = pd.DataFrame(feature)
        feature_df.to_csv(PREDICT_CSV, index=False)

        df = pd.read_csv(PREDICT_CSV)
        prediction = predict(df)

        print("Prediction:", prediction if prediction else "No match found.")
        if transcribed_text is not None:
            print("Sentence Match:", transcribed_text.lower() == sentence.lower())

def feature_extraction_new():
    """Extracts features from all stored voice samples."""
    data = []
    for speaker in os.listdir(VOICE_DATA_DIR):
        speaker_path = os.path.join(VOICE_DATA_DIR, speaker)
        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_path, file)
                    features, _ = extract_formants(file_path)
                    if features:
                        features["Label"] = speaker
                        data.append(features)
    df = pd.DataFrame(data)
    df.to_csv(FEATURES_CSV, index=False)
    print("Feature extraction completed.")                     

if __name__ == "__main__":
    choice = int(input("Choose an option: \n1. New Voice Input\n2. feature extraction \n3. Train Model\n4. Predict Voice\n->"))
    if choice == 1:
        new_input()
    elif choice == 2:
        feature_extraction_new()
    elif choice == 3:
        train_model(FEATURES_CSV)
    elif choice == 4:
        predict_voice()
    else:
        print("Invalid option.")