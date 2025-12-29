from .utils_audio import find_any_audio_file, plot_mfcc


def generate_mfcc_visualization():
    file_path = find_any_audio_file()

    if file_path is None:
        print("Error: No .wav files found in dataset.")
        return

    plot_mfcc(file_path)
