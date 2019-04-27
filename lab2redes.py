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

# Se grafica dominio del tiempo
plotFunction.figure('Dominio del tiempo Audio Original')
plotFunction.title('Dominio del tiempo Audio Original')
plotFunction.plot(audioTime, data)

# Se genera espectrograma del audio original
(frequencies, times, audioSpectogram) = signal.spectrogram(data, fs)

# Se genera el grafico
plotFunction.figure('Espectograma')
plotFunction.pcolormesh(times, frequencies, np.log10(audioSpectogram))
plotFunction.title("Espectrograma del audio original")
plotFunction.xlabel('Tiempo (S)')
plotFunction.ylabel('Frecuencia (Hz)')
plotFunction.colorbar()

# Se calcula al transformada de Fourier
fourierTransform = fourier(data)

# Se obtienen las frecuencias
freqs = frequency(len(data), 1/fs)

# Se genera el grafico
plotFunction.figure('Frecuencia audio original')
plotFunction.title('Dominio de la frecuencia audio original')
plotFunction.plot(freqs, fourierTransform)

# Se genera un filtro paso bajo
b, a = signal.butter(3, 0.05)

# Se aplica el filtro al audio
filtered = scipy.signal.lfilter(b, a, data)

# Se calcula la transformada de fourier del audio filtrado
fourierTransformFiltered = fourier(filtered)

# Se grafica el dominio de la frecuencia
plotFunction.figure('Dominio de la frecuencia Paso Bajo')
plotFunction.title('Dominio de la frecuencia Paso Bajo')
plotFunction.plot(freqs, fourierTransformFiltered)

# Se obtiene la transformada inversa
inverseFourierTransform = inverseFourier(fourierTransformFiltered)

plotFunction.figure('Dominio del tiempo Paso Bajo')
plotFunction.title('Dominio del tiempo Paso Bajo')
plotFunction.plot(audioTime, inverseFourierTransform)

(freqLowPass, timeLowPass, audioSpectogramLowPass) = signal.spectrogram(filtered, fs)

plotFunction.figure('Espectograma Paso Bajo')
plotFunction.pcolormesh(timeLowPass, freqLowPass, np.log10(audioSpectogramLowPass))
plotFunction.title("Espectograma Filtro Paso Bajo")
plotFunction.xlabel('Tiempo (s)')
plotFunction.ylabel('Frecuencia (Hz)')
plotFunction.colorbar()

# Se genera archivo de audio lowpass
finalSignal = np.asarray(inverseFourierTransform, dtype=np.int16)
wav.write('filtrado.wav', fs, finalSignal)

#Se muestran los graficos
plotFunction.show()