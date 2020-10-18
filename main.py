# -*- coding: utf-8 -*-
import scipy.io.wavfile as spiowf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import string


def texto(data):
    # print(data)
    # print(data.shape)
    # print(len(data))
    alfabeto = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # 26 + 26, min e maiusc
    ocorrencias = np.zeros(len(alfabeto))
    for caracter in data:
        print(ord(caracter))
        int_caracter = ord(caracter)
        if 65 <= int_caracter <= 90:
            numero_caracter = int_caracter % 65
            ocorrencias[numero_caracter] += 1
            print(f"Letra : {caracter}, {ocorrencias[numero_caracter]}")
        elif 97 <= int_caracter <= 122:
            numero_caracter = (int_caracter % 97) + 26
            ocorrencias[numero_caracter] += 1
            print(f"Letra : {caracter}, {ocorrencias[numero_caracter]}")

    histograma(alfabeto, ocorrencias)
    return ocorrencias


def img_audio(data):
    data_copy = data.flatten()
    tipo = str(data_copy.dtype)
    string_numero = ''
    for elemento in tipo:
        if elemento.isdigit():
            string_numero += elemento
    data_type = int(string_numero)

    alfabeto = np.arange(0, pow(2, data_type))
    ocorrencias = np.zeros(len(alfabeto))  # para termos um array com o tamanho ideal
    for element in data_copy:
        ocorrencias[element] += 1

    histograma(alfabeto, ocorrencias)
    return ocorrencias


def histograma(alfabeto, ocorrencias):
    print(ocorrencias)
    plt.bar(alfabeto, ocorrencias)
    plt.show()


def entropia(ocorrencias):
    entropia = 0
    probabilidade = np.zeros(len(ocorrencias))

    for i in range(len(ocorrencias)):
        probabilidade[i] = ocorrencias[i] / sum(ocorrencias)

        if probabilidade[i] != 0:
            entropia += (probabilidade[i] * np.log2(probabilidade[i]))

    return -entropia


def main(filename):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Se o ficheiro tiver uma destas extenções, significa que corresponde a uma imagem
        data = mpimg.imread(filename)
        if data.ndim == 4:
            ocorrencias = img_audio(data[:, :, :, 0])
        else:
            ocorrencias = img_audio(data)

    elif filename.lower().endswith(('.wav', '.mp3')):
        [sampleRate, data] = spiowf.read(filename)
        if data.ndim == 2:
            ocorrencias = img_audio(data[:, 0])
        else:
            ocorrencias = img_audio(data)

    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            data = np.asarray(list(f.read()))

        ocorrencias = texto(data)

    print(f"\nENTROPIA:\n {entropia(ocorrencias)}")


if __name__ == "__main__":
    #main("lena.bmp")
    main("binaria.bmp")
    #main("ct1.bmp")
    #main("texto.txt")
    #main("Song01.wav")
    #main("Saxriff.wav")
