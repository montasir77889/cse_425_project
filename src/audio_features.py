import librosa
import numpy as np
import os

def load_audio_mels(data_root, max_files=2000):
    features = []

    language_folders = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]

    print("Detected language folders:", language_folders)

    for lang in language_folders:
        lang_path = os.path.join(data_root, lang)

        for genre in os.listdir(lang_path):
            genre_path = os.path.join(lang_path, genre)
            if not os.path.isdir(genre_path):
                continue

            for file in os.listdir(genre_path):
                if file.lower().endswith(".wav"):
                    path = os.path.join(genre_path, file)
                    try:
                        y, sr = librosa.load(path, sr=22050)
                        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                        mel = librosa.power_to_db(mel)

                        # ✅ NORMALIZATION (CRITICAL)
                        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

                        mel = librosa.util.fix_length(mel, size=128, axis=1)
                        features.append(mel)
                    except:
                        continue

                    if len(features) >= max_files:
                        return np.array(features)

    return np.array(features)
