import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import cv2

song = '/home/victordev/Music/MeAndThoseDreaminEyesOfMine.wav'
acapella = '/home/victordev/Music/MeAndThoseDreaminEyesOfMine(Acapella).wav'

y, sr = librosa.load(song,sr=None, offset=15.0, duration=6.0)
y_a, sr_a = librosa.load(acapella,sr=None, offset=13.5, duration=6.0)

D = np.abs(librosa.stft(y))
D_a = np.abs(librosa.stft(y_a))

downsampled_y = librosa.resample(y, sr, 10880)

dsy_fourier = np.abs(librosa.stft(downsampled_y,n_fft=1022,win_length=1022, hop_length=256))
dsy_fourier = cv2.resize(dsy_fourier,(256,256),interpolation=cv2.INTER_AREA)

librosa.display.specshow(librosa.amplitude_to_db(dsy_fourier,ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

librosa.display.specshow(librosa.amplitude_to_db(D_a,ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram (acapella)')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
