
import numpy as np
import librosa

def extract_features(path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(librosa.power_to_db(mel).T, axis=0)

    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)

    feature_vector = np.hstack([mfccs_mean, chroma_mean, mel_mean[:10], contrast_mean, tonnetz_mean])
    return feature_vector
