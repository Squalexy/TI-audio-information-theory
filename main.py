import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as spiowf


#
def img_audio(data):
    data_copy = data.flatten()  # Convertemos num array unidimensional

    tipo = str(data_copy.dtype)  # Convertemos dtype para string
    string_numero = ''
    # Verifica o tipo de dados com dtype e extrai apenas o(s) digito(s) correspondente(s) aos bits
    for elemento in tipo:
        if elemento.isdigit():
            string_numero += elemento
    data_type = int(string_numero)  # Converte para inteiro a string correspondente aos bits

    alfabeto = np.arange(0, pow(2, data_type))   # Criamos um array do alfabeto que vai de 0 a 2^data_type
    ocorrencias = np.zeros(len(alfabeto))  # Criamos um array com o numero de ocorrencias de cada elemento do alfabeto

    #
    for element in data_copy:
        ocorrencias[element] += 1

    histograma(alfabeto, ocorrencias)

    return ocorrencias


def texto(data):
    alfabeto = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # 26 + 26, min e maiusc
    ocorrencias = np.zeros(len(alfabeto))

    for caracter in data:
        int_caracter = ord(caracter)

        if 65 <= int_caracter <= 90:
            numero_caracter = int_caracter % 65
            ocorrencias[numero_caracter] += 1

        elif 97 <= int_caracter <= 122:
            numero_caracter = (int_caracter % 97) + 26
            ocorrencias[numero_caracter] += 1

    histograma(alfabeto, ocorrencias)

    return ocorrencias


def histograma(alfabeto, ocorrencias):
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
    # Verifica se é um ficheiro imagem
    if filename.lower().endswith('.bmp'):  # Extensões correspondentes a imagem
        data = mpimg.imread(filename)

        # Se for uma imagem com 3 dimensões(a cores), usamos apenas o channel vermelho
        if data.ndim == 3:
            ocorrencias = img_audio(data[:, :, 0])

        else:
            ocorrencias = img_audio(data)

    # Verifica se é um ficheiro áudio
    elif filename.lower().endswith(('.wav', '.mp3')):
        [sampleRate, data] = spiowf.read(filename)

        # Se for um audio stereo usamos apenas o canal esquerdo
        if data.ndim == 2:
            ocorrencias = img_audio(data[:, 0])
        else:
            ocorrencias = img_audio(data)

    # Verifica se é um ficheiro texto
    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            # Convertemos uma matriz de linhas para um array unidimensional só com chars
            data = np.asarray(list(f.read()))

        ocorrencias = texto(data)

    # Imprime entropia - limite mínimo teórico para o número médio de bits por símbolo
    print(f"Entropia: {entropia(ocorrencias)}")


if __name__ == "__main__":
    # -----IMAGENS-----
    main("lena.bmp")
    # main("binaria.bmp")
    # main("ct1.bmp")

    # -----TEXTO-----
    # main("texto.txt")

    # -----AUDIO-----
    # main("saxriff.wav")
    # main("Song01.wav")
    # main("Song02.wav")
    # main("Song03.wav")
    # main("Song04.wav")
    # main("Song05.wav")
    # main("Song06.wav")
    # main("Song07.wav")
    # main("target01 - repeat.wav")
    # main("target02 - repeatNoise.wav")
