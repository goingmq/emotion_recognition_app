import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt

from .config import DATA_PATH


def find_any_audio_file():
    search = os.path.join(str(DATA_PATH), "**", "*.wav")
    files = glob.glob(search, recursive=True)

    if not files:
        return None

    return files[0]


def plot_mfcc(file_path, output="mfcc_spectrogram.png", sr=22050, n_mfcc=40):
    print(f"Generating MFCC for: {file_path}")

    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr, cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()

    plt.savefig(output, dpi=300)
    plt.show()

    print(f"Saved MFCC spectrogram to {output}")
