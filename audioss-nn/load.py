import numpy as np
import musdb
import librosa.display
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras

""" convert to mono -> downsample to 10880hz -> 
    split into 6s chunks (Hanning window size 1022 and hop-size 256) total of 512x256 spectograms -> 
    resample to 256x256 spectograms"""

def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image

def audio_to_spectogram(audio, sr):
    mono = to_mono(audio)
    ds_audio = downsample_audio_to_10880hz(mono, sr)
    spectogram = stft(ds_audio)
    spectogram = normalize(spectogram)
    # return downsample_spectogram(spectogram)
    return spectogram

def to_mono(audio):
    return librosa.to_mono(audio)

def downsample_audio_to_10880hz(audio_mono, sr):
    return librosa.resample(audio_mono, sr, 10880)

"""get spectogram with a Hanning window size of 1022 and hop-size of 256"""
def stft(audio):
    return np.abs(librosa.stft(audio, n_fft=1022, win_length=1022, hop_length=256))

# """resample to 256x256 spectogram"""
# def downsample_spectogram(spectogram):
#     return cv2.resize(spectogram, (256, 256), interpolation=cv2.INTER_AREA)

def plot_spectogram(D, is_track=True):
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    if is_track:
        plt.title('Track spectrogram')
    else:
        plt.title('Vocals spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mus_train = musdb.DB(root="musdb", subsets="train")
    mus_test = musdb.DB(root="musdb", subsets="test")

    for track in mus_train.tracks:
        track_stereo = track.audio.T
        start = 0
        track.chunk_duration = 6.0
        sample_rate = track.rate

        while start < track.duration:
            track.chunk_start = start

            track_stereo = track.audio.T
            vocals_stereo = track.targets['vocals'].audio.T

            track_spectogram = audio_to_spectogram(track_stereo, sample_rate)
            vocals_spectogram = audio_to_spectogram(vocals_stereo, sample_rate)

            start = start + track.chunk_duration

            plot_spectogram(track_spectogram)
            plot_spectogram(vocals_spectogram, is_track=False)
