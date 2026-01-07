import os
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_len=130):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

def build_dataset(data_root):
    X = []
    for lang in ["Bangla_Datasets", "English_Datasets"]:
        for genre in os.listdir(os.path.join(data_root, lang)):
            genre_path = os.path.join(data_root, lang, genre)
            for f in os.listdir(genre_path):
                if f.endswith(".wav"):
                    X.append(extract_mfcc(os.path.join(genre_path, f)))
    return np.array(X)
