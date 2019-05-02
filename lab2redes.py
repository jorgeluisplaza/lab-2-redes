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

# Recibe las amplitudes, los tiempos y las frecuencias y genera los graficos y Espectrograma del audio original
def originalAudio(data, audioTime, freqs):

    # Se calcula al transformada de Fourier
    fourierTransform = fourier(data)

    # Se grafica dominio del tiempo
    generatePlot('Dominio del tiempo Audio Original', 'Dominio del tiempo Audio Original', audioTime, data)

    # Se genera grafico en frecuencia
    generatePlot('Frecuencia audio original', 'Dominio de la frecuencia audio original',freqs, fourierTransform)

    # Se genera Espectrograma
    generateSpectogram('Espectograma', "Espectrograma del audio original", 'Tiempo (S)', 'Frecuencia (Hz)', data, fs)

# Recibe las amplitudes, la freq de muestreo, los tiempos y las frecuencias
# Y realizar un filtro FIR de Paso Bajo generando graficos y Espectrograma
def FIRFilterLowPass(data, fs, audioTime, freqs):
    # Se genera un filtro paso bajo
    b = signal.firwin(50, 1500, fs=fs)

    # Se aplica el filtro FIR al audio
    filtered = scipy.signal.lfilter(b, [1.0], data)

    # Se calcula la transformada de fourier del audio filtrado
    fourierTransformFiltered = fourier(filtered)

    # Se grafica el dominio de la frecuencia
    generatePlot('Dominio de la frecuencia Paso Bajo', 'Dominio de la frecuencia Paso Bajo', freqs, fourierTransformFiltered)

    # Se obtiene la transformada inversa
    inverseFourierTransform = inverseFourier(fourierTransformFiltered)

    # Se grafica en el dominio del tiempo
    generatePlot('Dominio del tiempo Paso Bajo', 'Dominio del tiempo Paso Bajo', audioTime, inverseFourierTransform)

    # Se genera el Espectograma
    generateSpectogram('Espectrograma Paso Bajo', "Espectrograma Filtro Paso Bajo", 'Tiempo (s)', 'Frecuencia (Hz)', filtered, fs)

    # Se genera archivo de audio lowpass
    finalSignal = np.asarray(inverseFourierTransform, dtype=np.int16)
    wav.write('lowPassFilter.wav', fs, finalSignal)

# Recibe las amplitudes, la freq de muestreo, los tiempos y las frecuencias
# Y realiza un filtro FIR de Paso Alto generando graficos y Espectrograma
def FIRFilterHighPass(data, fs, audioTime, freqs):
    # Se genera filtro pasa alto
    highPassFilter = signal.firwin(49, 1500, fs=fs, pass_zero=False)

    # Se aplica el filtro FIR al audio
    filteredHighPass = scipy.signal.lfilter(highPassFilter, [1.0], data)

    # Se calcula su transformada de Fourier
    highPassFourierTransform = fourier(filteredHighPass)

    # Se grafica en el dominio de la frecuencia
    generatePlot('Dominio de la frecuencia Pasa Alto', 'Dominio de la frecuencia Pasa Alto', freqs, highPassFourierTransform)

    # Se obtiene la transformada inversa
    inverseHighPassFourierTransform = inverseFourier(highPassFourierTransform)

    # Se grafica en el dominio del tiempo
    generatePlot('Dominio del tiempo Pasa Alto', 'Dominio del tiempo Pasa Alto', audioTime, inverseHighPassFourierTransform)

    # Se genera Espectrograma
    generateSpectogram('Espectrograma Pasa Alto', 'Espectrograma Pasa Alto', 'Tiempo (s)', 'Frecuencia (Hz)', filteredHighPass, fs)

    finalSignal = np.asarray(inverseHighPassFourierTransform, dtype=np.int16)
    wav.write('highPassFilter.wav', fs, finalSignal)

# Recibe las amplitudes, la freq de muestreo, los tiempos y las frecuencias
# Y realiza un filtro FIR de Pasa Banda generando graficos y Espectrograma
def FIRFilterBandPass(data, fs, audioTime, freqs):
    b = signal.firwin(49, [500, 2000], fs = fs)

    filtered = scipy.signal.lfilter(b, [1.0], data)

    fourierTransformFiltered = fourier(filtered)

    generatePlot('Dominio de la frecuencia Pasa Banda', 'Dominio de la frecuencia Pasa Banda', freqs, fourierTransformFiltered)

    inverseFourierTransform = inverseFourier(fourierTransformFiltered)

    generatePlot('Dominio del tiempo Pasa Banda', 'Dominio del tiempo Pasa Banda', audioTime, inverseFourierTransform)

    generateSpectogram('Espectograma Pasa Banda', 'Espectograma Filtro Pasa Banda', 'Tiempo (s)', 'Frecuencia (Hz)', filtered, fs)

    finalSignal = np.asarray(inverseFourierTransform, dtype=np.int16)
    wav.write('bandPassFilter.wav', fs, finalSignal)

# Recibe un nombre de figura, un titulo y nombres para eje x y eje y
# Y genera un grafico con sus valores
def generatePlot(figName, figTitle, xlab, ylab):
    plotFunction.figure(figName)
    plotFunction.title(figTitle)
    plotFunction.plot(xlab, ylab)

# Recibe un nombre de figura, un titulo y nombres para eje x y eje y
# las amplitudes y la freq de muestreo y genera un Espectrograma
def generateSpectogram(figName, figtitle, xlabel, ylabel, data, fs):
    # Se genera espectrograma del audio original
    (frequencies, times, audioSpectogram) = signal.spectrogram(data, fs)

    # Se genera grafico
    plotFunction.figure(figName)
    plotFunction.pcolormesh(times, frequencies, np.log10(audioSpectogram))
    plotFunction.title(figtitle)
    plotFunction.xlabel(xlabel)
    plotFunction.ylabel(ylabel)
    plotFunction.colorbar()


# Se lee el archivo de audio handel y se guarda
# fs es la frecuencia de muestreo
# data es el valor de la amplitudes
(fs, data) = wav.read('handel.wav')

# Se genera los valores de tiempo entre cada amplitud
audioTime = np.linspace(0, len(data)/fs, num=len(data))

# Se obtienen las frecuencias
freqs = frequency(len(data), 1.0/fs)

originalAudio(data, audioTime, freqs)
FIRFilterLowPass(data, fs, audioTime, freqs)
FIRFilterHighPass(data, fs, audioTime, freqs)
FIRFilterBandPass(data, fs, audioTime, freqs)

#Se muestran los graficos
plotFunction.show()