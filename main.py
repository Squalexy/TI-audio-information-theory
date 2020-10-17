import scipy.io.wavfile as spiowf
import sounddevice as sd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def histograma(data):
    # print(data)
    # print(data.shape)
    # print(len(data))
    data_copy = data.flatten()

    if data_copy.dtype == 'str_':
        for caracter in data_copy:
            int_caracter = int(caracter)
            if 65 <= int_caracter <= 90:
                numero_caracter = int_caracter % 65
            elif 97 <= int_caracter <= 122:
                numero_caracter = (int_caracter % 97) + 26


    ocorrencias = np.zeros(len(max(data_copy)))  # para termos um array com o tamanho ideal
    for elem in data_copy:
        ocorrencias[elem] += 1


def main():
    # filename = input("Nome do ficheiro : ")
    filename = "lena.bmp"

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        data = mpimg.imread(filename)

    elif filename.lower().endswith(('.wav', '.mp3')):
        [samplerate, data] = spiowf.read(filename)

    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            data = f.read()
            data = data.split('\n')

    histograma(data)


if __name__ == "__main__":
    main()
