import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as spiowf
from huffmancodec import HuffmanCodec

'''
FUNÇÃO para criar alfabeto e nº de ocorrências de ficheiros audio e imagem
O nº de ocorrências é obtido através da data e do seu alfabeto
Através do dtype obtemos o tamanho necessário para criar o array onde guardamos o nº de ocorrências
A função mostra o histograma da data recebida e retorna o nº de ocorrências a ser usado na função da entropia
'''


def img_audio(data):
    data_copy = data.flatten()

    tipo = str(data_copy.dtype)
    string_numero = ''
    for elemento in tipo:
        if elemento.isdigit():
            string_numero += elemento
    data_type = int(string_numero)

    alfabeto = np.arange(0, 2 ** data_type)
    ocorrencias = np.zeros(len(alfabeto))

    for element in data_copy:
        ocorrencias[element] += 1

    histograma(alfabeto, ocorrencias)

    return ocorrencias, alfabeto, data_copy


'''
FUNÇÃO para criar alfabeto e nº de ocorrências de ficheiros texto
O alfabeto que criamos tem os caracteres "A" a "Z" + "a" a "z" (52 caracteres no total)
O nº de ocorrências é obtido através da data e do seu alfabeto
A função mostra o histograma da data recebida e retorna o nº de ocorrências a ser usado na função da entropia
'''


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

    return ocorrencias, alfabeto


'''FUNÇÃO que cria um histograma consoante o alfabeto recebido e o nº de ocorrências de cada símbolo do alfabeto'''


def histograma(alfabeto, ocorrencias):
    plt.bar(alfabeto, ocorrencias)
    plt.show()


'''FUNÇÃO para calcular a entropia de determinada data, através do nº de ocorrências do alfabeto dessa data'''


def calc_entropia(ocorrencias):
    probabilidade = np.zeros(len(ocorrencias))

    for i in range(len(ocorrencias)):
        probabilidade[i] = ocorrencias[i] / sum(ocorrencias)

    mask = np.isin(probabilidade, 0, invert=True)
    probabilidade = probabilidade[mask]
    entropia = np.sum(probabilidade * np.log2(probabilidade))

    return -entropia, probabilidade


def media_ponderada(alfabeto, data, flag, probabilidade):

    """
    Retirar caracteres e elementos desnecessários que não façam parte do alfabeto
    """

    if flag == 1:
        data_copy = []
        for i in range(len(data)):
            if data[i] in alfabeto:
                data_copy += data[i]
        data = np.array(data_copy)

    codec = HuffmanCodec.from_data(data)
    t = codec.get_code_table()

    """
    Os elementos que não ocorrem já foram retirados com o HuffmanCodec
    """

    if flag == 1:
        delete = [key for key in t if key not in alfabeto]
        for key in delete:
            del t[key]

    s, l = codec.get_code_len()
    print(t)
    print(s)
    print(l)
    print(probabilidade)

    print(len(s))
    print(len(l))
    print(len(probabilidade))


    #calc_media_sum = np.sum(probabilidade * l)
    #print(f"NP SUM: {calc_media_sum}")

    calc_media_ponderada = np.average(l, axis=None, weights=probabilidade)

    print(f"Média ponderada: {calc_media_ponderada}")

    return calc_media_ponderada, l


def variancia_ponderada(probabilidade, calc_media_ponderada, l):
    calc_media_ponderada_quadrado = np.average(l ** 2, axis=None, weights=probabilidade)
    calc_variancia_ponderada = calc_media_ponderada_quadrado - calc_media_ponderada ** 2
    print(f"Variância ponderada: {calc_variancia_ponderada}")


def main(filename):
    # Condições para verificar extensão do ficheiro, afim de chamar a função correspondente

    flag = 0  # Condição para a função média_ponderada, dependendo se for txt ou img/audio

    # Condição para imagens
    if filename.lower().endswith('.bmp'):
        data = mpimg.imread(filename)

        # Se for uma imagem com 3 dimensões(a cores), usamos apenas o canal vermelho
        if data.ndim == 3:
            ocorrencias, alfabeto, data = img_audio(data[:, :, 0])

        else:
            ocorrencias, alfabeto, data = img_audio(data)

    # Condição para audios
    elif filename.lower().endswith(('.wav', '.mp3')):
        [sampleRate, data] = spiowf.read(filename)

        # Se for um audio stereo usamos apenas o canal esquerdo
        if data.ndim == 2:
            ocorrencias, alfabeto, data = img_audio(data[:, 0])
        else:
            ocorrencias, alfabeto, data = img_audio(data)

    # Condição para textos
    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            data = np.asarray(list(f.read()))
        flag = 1
        ocorrencias, alfabeto = texto(data)

    entropia, probabilidade = calc_entropia(ocorrencias)

    print(f"Entropia: {entropia}")  # resultado[0] = ocorrências
    calc_media_ponderada, length = media_ponderada(alfabeto, data, flag, probabilidade)
    variancia_ponderada(probabilidade, calc_media_ponderada, length)


if __name__ == "__main__":
    # -----IMAGENS-----
    # main("lena.bmp")
    # main("binaria.bmp")
    # main("ct1.bmp")

    # -----TEXTO-----
    main("texto.txt")

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
