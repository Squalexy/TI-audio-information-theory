import scipy.io.wavfile as spiowf
import sounddevice as sd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import string


def histograma(data):
    # print(data)
    # print(data.shape)
    # print(len(data))

    if data[0].isalpha():
        alfabeto =  list(string.ascii_uppercase) + list(string.ascii_lowercase)  # 26 + 26, min e maiusc
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
    else:
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


    print(ocorrencias)
    plt.bar(alfabeto, ocorrencias)
    plt.show()



def main():
    # filename = input("Nome do ficheiro : ")
    filename = "lena.bmp"

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        data = mpimg.imread(filename)

    elif filename.lower().endswith(('.wav', '.mp3')):
        [samplerate, data] = spiowf.read(filename)

    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            data = list(f.read())

    # print(ord(data[0]))
    histograma(data)


if __name__ == "__main__":
    main()
