import numpy as np
import scipy
import matplotlib
import scipy.io.wavfile as wav
import matplotlib.pyplot as plotFunction
from scipy.fftpack import fft as fourier
from scipy.fftpack import ifft as inverseFourier 
from scipy.fftpack import fftfreq as frequency
from scipy import signal
import pylab
import math

# Se lee el archivo de audio handel y se guarda
# fs es la frecuencia
# data es el valor de la amplitudes
(fs, data) = wav.read('handel.wav')

# Se genera los valores de tiempo entre cada amplitud
audioTime = np.linspace(0, len(data)/fs, num=len(data))

(frequencies, times, audioSpectogram) = signal.spectrogram(data, fs)

plotFunction.pcolormesh(times, frequencies, np.log10(audioSpectogram))
plotFunction.xlabel('Tiempo (S)')
plotFunction.ylabel('Frecuencia (Hz)')
plotFunction.colorbar()

fourierTransform = fourier(data)

freqs = frequency(len(data), 1.0/fs)

cutOffFreq = 400.0
freqRatio = (cutOffFreq/fs)
N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)
win = np.ones(N)
win *= 1.0/N
filtered = scipy.signal.lfilter(win, [1], data)

fourierTransformFiltered = fourier(filtered)

plotFunction.figure('Grafico 1')
plotFunction.plot(freqs, fourierTransform)
plotFunction.figure('Grafico 2')
plotFunction.plot(freqs, fourierTransformFiltered)

inverseFourierTransform = np.asarray(filtered, dtype=np.int16)

wav.write('filtered.wav', freqs, inverseFourierTransform)

#plotFunction.plot(filtered)

#Se muestran los graficos
plotFunction.show()