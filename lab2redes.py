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
b = signal.firwin(50, 1500, fs=fs)

# Se aplica el filtro FIR al audio
filtered = scipy.signal.lfilter(b, [1.0], data)

# Se calcula la transformada de fourier del audio filtrado
fourierTransformFiltered = fourier(filtered)

# Se grafica el dominio de la frecuencia
plotFunction.figure('Dominio de la frecuencia Paso Bajo')
plotFunction.title('Dominio de la frecuencia Paso Bajo')
plotFunction.plot(freqs, fourierTransformFiltered)

# Se obtiene la transformada inversa
inverseFourierTransform = inverseFourier(fourierTransformFiltered)

# Se grafica en el dominio del tiempo
plotFunction.figure('Dominio del tiempo Paso Bajo')
plotFunction.title('Dominio del tiempo Paso Bajo')
plotFunction.plot(audioTime, inverseFourierTransform)

# Se obtiene su Espectograma
(freqLowPass, timeLowPass, audioSpectogramLowPass) = signal.spectrogram(filtered, fs)

# Se grafica el Espectograma
plotFunction.figure('Espectrograma Paso Bajo')
plotFunction.pcolormesh(timeLowPass, freqLowPass, np.log10(audioSpectogramLowPass))
plotFunction.title("Espectrograma Filtro Paso Bajo")
plotFunction.xlabel('Tiempo (s)')
plotFunction.ylabel('Frecuencia (Hz)')
plotFunction.colorbar()

# Se genera archivo de audio lowpass
finalSignal = np.asarray(inverseFourierTransform, dtype=np.int16)
wav.write('filtrado.wav', fs, finalSignal)

# Se genera filtro pasa alto
highPassFilter = signal.firwin(49, 1500, fs=fs, pass_zero=False)

# Se aplica el filtro FIR al audio
filteredHighPass = scipy.signal.lfilter(highPassFilter, [1.0], data)

# Se obtiene su Espectrograma
(freqHighPass, timeHighPass, audioSpectogramHighPass) = signal.spectrogram(filteredHighPass, fs)

# Se grafica el Espectrograma
plotFunction.figure('Espectrograma Pasa Alto')
plotFunction.pcolormesh(timeHighPass, freqHighPass, np.log10(audioSpectogramHighPass))
plotFunction.title('Espectrograma Pasa Alto')
plotFunction.xlabel('Tiempo (s)')
plotFunction.ylabel('Frecuencia (Hz)')
plotFunction.colorbar()

# Se calcula su transformada de Fourier
highPassFourierTransform = fourier(filteredHighPass)

# Se grafica en el dominio de la frecuencia
plotFunction.figure('Dominio de la frecuencia Pasa Alto')
plotFunction.title('Dominio de la frecuencia Pasa Alto')
plotFunction.plot(freqs, highPassFourierTransform)

# Se obtiene la transformada inversa
inverseHighPassFourierTransform = inverseFourier(highPassFourierTransform)

# Se grafica en el dominio del tiempo
plotFunction.figure('Dominio del tiempo Pasa Alto')
plotFunction.title('Dominio del tiempo Pasa Alto')
plotFunction.plot(audioTime, inverseHighPassFourierTransform)

#Se muestran los graficos
plotFunction.show()