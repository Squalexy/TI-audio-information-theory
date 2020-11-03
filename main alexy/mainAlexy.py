import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as spiowf
from huffmancodec import HuffmanCodec

'''
FUNÇÃO que transforma o tipo de um ficheiro (ex: uint8) em digito (ex: 8) para obtermos o tamanho da data desse ficheiro.
'''


def retira_tipo_dados(data):
    tipo = str(data.dtype)
    string_numero = ''
    for elemento in tipo:
        if elemento.isdigit():
            string_numero += elemento
    data_type = int(string_numero)

    return data_type


'''
FUNÇÃO que retorna um array com o nº de ocorrências de cada símbolo de um ficheiro num determinado alfabeto.
O array contém na mesma as ocorrências a 0.
'''


def cria_ocorrencias(alfabeto, data):
    ocorrencias = np.zeros(len(alfabeto))
    for element in data:
        ocorrencias[element] += 1
    return ocorrencias


'''
Retira os caracteres especiais do texto (ex: @, é, etc)
'''


def retirar_texto(data):
    alfabeto = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    data_copy = []
    for i in range(len(data)):
        if data[i] in alfabeto:
            data_copy += data[i]
    data = np.array(data_copy)
    return data, alfabeto


'''
Converte caracteres de um txt em números correspondentes ao índice desse caracter
'''


def cria_array_numeros(data):
    array_numeros = []
    for caracter in data:
        int_caracter = ord(caracter)

        if 65 <= int_caracter <= 90:
            array_numeros.append(int_caracter % 65)

        elif 97 <= int_caracter <= 122:
            array_numeros.append((int_caracter % 97) + 26)

    return array_numeros


'''
FUNÇÃO para criar histograma de um ficheiro txt e retorna o nº de ocorrências de cada símbolo
'''


def texto(array_numeros, alfabeto):
    ocorrencias = cria_ocorrencias(alfabeto, array_numeros)
    histograma(alfabeto, ocorrencias)
    return ocorrencias


'''
FUNÇÃO que retorna o nº de ocorrências, alfabeto e data de um ficheiro áudio e imagem;
Serve para calcular a entropia e mostrar o histograma desse ficheiro.
'''


def img_audio(data):
    data_copy = data.flatten()
    data_type = retira_tipo_dados(data_copy)

    alfabeto = np.arange(0, 2 ** data_type)
    ocorrencias = cria_ocorrencias(alfabeto, data_copy)

    return ocorrencias, alfabeto, data_copy


'''FUNÇÃO que cria um histograma consoante o alfabeto recebido e o nº de ocorrências de cada símbolo do alfabeto'''


def histograma(alfabeto, ocorrencias):
    plt.bar(alfabeto, ocorrencias)
    plt.show()


'''
FUNÇÃO para calcular a entropia de determinada data, através do nº de ocorrências do alfabeto dessa data
'''


def calc_probabilidade(ocorrencias):
    probabilidade = np.zeros(len(ocorrencias))
    sum_ocorrencias = sum(ocorrencias)
    for i in range(len(ocorrencias)):
        if ocorrencias[i] != 0:
            probabilidade[i] = ocorrencias[i] / sum_ocorrencias
    return probabilidade


"""
FUNÇÃO para calcular a probabilidade conjunta entre o query e target numa janela
"""


def calc_probabilidade_conjunta(query, target, alfabeto, passo):
    ocorrencia_conjunta = np.zeros((len(alfabeto), len(alfabeto)))
    # Adicionar na matriz +1 ocorrências relativamente aos valores do query e do target
    # O valor do query dá-nos a linha e o valor do target dá-nos a coluna
    for j in range(len(query)):
        linha = query[j]
        coluna = target[j]
        ocorrencia_conjunta[linha][coluna] += 1
    # Criar o array de probabilidades das ocorrências entre o query e o target
    probabilidade_conjunta = ocorrencia_conjunta / len(query)
    return probabilidade_conjunta


def calc_entropia(ocorrencias):
    probabilidade = calc_probabilidade(ocorrencias)
    mask = np.isin(probabilidade, 0, invert=True)
    probabilidade = probabilidade[mask]
    entropia = np.sum(probabilidade * np.log2(probabilidade))
    return -entropia, probabilidade


'''
FUNÇÃO para calcular média ponderada de uma determinada fonte usando cod HUFFMAN
'''


def media_ponderada(data, probabilidade):
    codec = HuffmanCodec.from_data(data)
    t = codec.get_code_table()
    s, l = codec.get_code_len()

    calc_media_ponderada = np.average(l, axis=None, weights=probabilidade)
    print(f"Média ponderada: {calc_media_ponderada}")
    return calc_media_ponderada, l


'''
FUNÇÃO para calcular variância ponderada de uma determinada fonte usando cod HUFFMAN
'''


def variancia_ponderada(probabilidade, calc_media_ponderada, l):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    calc_media_ponderada_quadrado = np.average(l, axis=None, weights=probabilidade)
    calc_variancia_ponderada = calc_media_ponderada_quadrado - calc_media_ponderada ** 2
    print(f"Variância ponderada: {calc_variancia_ponderada}")


'''
FUNÇÃO para agrupar pares de símbolos de determinada data
'''


def agrupar_simbolos(data, tam_alfabeto):
    dim_data = tam_alfabeto * tam_alfabeto
    alfabeto = np.arange(0, dim_data)
    data_indices = []
    for i in range(0, len(data), 2):
        if i == len(data) - 1 and len(data) % 2 != 0:
            break
        else:
            indice = data[i] * tam_alfabeto + data[i + 1]
            data_indices.append(indice)
    ocorrencias = cria_ocorrencias(alfabeto, data_indices)
    entropia, probabilidade = calc_entropia(ocorrencias)
    entropia = entropia / 2
    print(f"Entropia pares: {entropia}")


def cria_info_mutua(query, target, alfabeto, passo):
    num_janelas = int(((len(target) - len(query)) / passo)) + 1
    inf_mutua = np.zeros(num_janelas)
    count = 0
    for i in range(0, len(target), passo):
        if i + len(query) <= len(target):
            ocorrencias_query = cria_ocorrencias(alfabeto, query)
            ocorrencias_target = cria_ocorrencias(alfabeto, target[i: i + len(query)])
            soma = 0
            prob_query = calc_probabilidade(ocorrencias_query)
            prob_target = calc_probabilidade(ocorrencias_target)
            prob_conjunta = calc_probabilidade_conjunta(query, target[i: i + len(query)], alfabeto, i)
            for linha in range(len(alfabeto)):
                for coluna in range(len(alfabeto)):
                    if prob_conjunta[linha][coluna] != 0 and prob_query[linha] != 0 and prob_target[coluna] != 0:
                        soma += (prob_conjunta[linha][coluna] * np.log2(prob_conjunta[linha][coluna] / (
                                prob_query[linha] * prob_target[coluna])))
            inf_mutua[count] = soma
            count += 1
    print(f"Informação mútua:  {inf_mutua}")


def main():
    # def main(filename):
    # Condições para verificar extensão do ficheiro, afim de chamar a função correspondente

    # Condição para imagens

    """
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
        data_letras, alfabeto = retirar_texto(data)

        print(data_letras)
        data = cria_array_numeros(data_letras)
        ocorrencias = texto(data, alfabeto)

    entropia, probabilidade = calc_entropia(ocorrencias)
    print(f"Nome do ficheiro: ")
    print(f"Entropia: {entropia}")

    calc_media_ponderada, length = media_ponderada(data, probabilidade)
    variancia_ponderada(probabilidade, calc_media_ponderada, length)
    agrupar_simbolos(data, len(alfabeto))

    """
    query = [2, 6, 4, 10, 5, 9, 5, 8, 0, 8]
    target = [6, 8, 9, 7, 2, 4, 9, 9, 4, 9,
              1, 4, 8, 0, 1, 2, 2, 6, 3, 2,
              0, 7, 4, 9, 5, 4, 8, 5, 2, 7,
              8, 0, 7, 4, 8, 5, 7, 4, 3, 2,
              2, 7, 3, 5, 2, 7, 4, 9, 9, 6]
    alfabeto = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    passo = 1
    cria_info_mutua(query, target, alfabeto, passo)


if __name__ == "__main__":
    main()
    # -----IMAGENS-----
    # main("lena.bmp")
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
