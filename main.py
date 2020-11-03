import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as spiowf
from huffmancodec import HuffmanCodec

'''----------------FUNÇÕES GERAIS----------------'''

'''
FUNÇÃO que retorna a data de um ficheiro áudio apenas num canal
'''


def extrair_audio(filename):
    [sampleRate, data] = spiowf.read(filename)
    if data.ndim == 2:
        ocorrencias, alfabeto, data = img_audio(data[:, 0])
    else:
        ocorrencias, alfabeto, data = img_audio(data)
    return ocorrencias, alfabeto, data


'''
FUNÇÃO que 
- transforma o tipo de um ficheiro (ex: uint8) em digito (ex: 8) 
- serve para obtermos o tamanho da data desse ficheiro
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


'''----------------EX 1----------------'''

'''
FUNÇÃO que retorna um array com o nº de ocorrências de cada símbolo de um ficheiro num determinado alfabeto
'''


def cria_ocorrencias(alfabeto, data):
    ocorrencias = np.zeros(len(alfabeto))
    for element in data:
        ocorrencias[element] += 1
    return ocorrencias


'''
FUNÇÃO que 
- retorna o nº de ocorrências de um ficheiro txt
'''


def texto(array_numeros, alfabeto):
    ocorrencias = cria_ocorrencias(alfabeto, array_numeros)
    return ocorrencias


'''
FUNÇÃO que retorna o nº de ocorrências, alfabeto e data de um ficheiro áudio e imagem
'''


def img_audio(data):
    data_copy = data.flatten()
    data_type = retira_tipo_dados(data_copy)

    alfabeto = np.arange(0, 2 ** data_type)
    ocorrencias = cria_ocorrencias(alfabeto, data_copy)

    return ocorrencias, alfabeto, data_copy


'''
FUNÇÃO que cria um histograma consoante o alfabeto recebido e o nº de ocorrências de cada símbolo do alfabeto
'''


def histograma(alfabeto, ocorrencias):
    plt.bar(alfabeto, ocorrencias)
    plt.show()


'''----------------EX 2 e 3----------------'''

'''
FUNÇÃO que calcula a probabilidade de cada símbolo ocorrer em determinado alfabeto
'''


def calc_probabilidade(ocorrencias):
    probabilidade = np.zeros(len(ocorrencias))
    sum_ocorrencias = sum(ocorrencias)
    for i in range(len(ocorrencias)):
        if ocorrencias[i] != 0:
            probabilidade[i] = ocorrencias[i] / sum_ocorrencias
    return probabilidade


'''
FUNÇÃO que calcula a entropia de determinada data, através do nº de ocorrências do alfabeto dessa data
'''


def calc_entropia(ocorrencias):
    probabilidade = calc_probabilidade(ocorrencias)
    mask = np.isin(probabilidade, 0, invert=True)
    probabilidade = probabilidade[mask]
    entropia = np.sum(probabilidade * np.log2(probabilidade))
    return -entropia, probabilidade


'''----------------EX 4----------------'''

'''
FUNÇÃO que calcula a média ponderada de uma determinada fonte usando cod HUFFMAN
'''


def calc_media_ponderada(data, probabilidade):
    codec = HuffmanCodec.from_data(data)
    s, l = codec.get_code_len()

    media_ponderada = np.average(l, axis=None, weights=probabilidade)
    print(f"Média ponderada: {media_ponderada}")
    return media_ponderada, l


'''
FUNÇÃO que calcula a variância ponderada de uma determinada fonte usando cod HUFFMAN
'''


def calc_variancia_ponderada(probabilidade, media_ponderada, l):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    calc_media_ponderada_quadrado = np.average(l, axis=None, weights=probabilidade)
    variancia_ponderada = calc_media_ponderada_quadrado - media_ponderada ** 2
    print(f"Variância ponderada: {variancia_ponderada}")


'''----------------EX 5----------------'''

'''
FUNÇÃO que
- agrupa em pares cada símbolo de um alfabeto
- calcula a entropia usando cod HUFFMAN
'''


def agrupar_simbolos(data, tam_alfabeto):
    dim_data = tam_alfabeto * tam_alfabeto
    alfabeto = np.arange(0, dim_data)
    data_indices = []
    for i in range(0, len(data), 2):
        # condição para se a data tiver comprimento ímpar e chegarmos ao último elemento sem par
        if i == len(data) - 1 and len(data) % 2 != 0:
            break
        else:
            indice = data[i] * tam_alfabeto + data[i + 1]
            data_indices.append(indice)
    ocorrencias = cria_ocorrencias(alfabeto, data_indices)
    entropia, probabilidade = calc_entropia(ocorrencias)
    entropia = entropia / 2
    print(f"Entropia pares: {entropia}")


'''----------------EX 6----------------'''

'''
FUNÇÃO que calcula o nº de janelas consoante o passo definido pelo utilizador
'''


def cria_num_janelas(query, target, passo):
    return int(((len(target) - len(query)) / passo)) + 1


'''
FUNÇÃO que calcula a probabilidade conjunta de 2 fontes: o query e o target
'''


def calc_probabilidade_conjunta(query, target, alfabeto):
    ocorrencia_conjunta = np.zeros((len(alfabeto), len(alfabeto)))
    for j in range(len(query)):
        ocorrencia_conjunta[query[j]][target[j]] += 1
    probabilidade_conjunta = ocorrencia_conjunta / len(query)

    return probabilidade_conjunta


'''
FUNÇÃO que 
- calcula a informação mútua de 2 fontes numa determinada janela
- usa apenas 1 ciclo
'''


def cria_informacao_cada_janela(query, target, alfabeto):
    prob_x = calc_probabilidade(cria_ocorrencias(alfabeto, query))
    prob_y = calc_probabilidade(cria_ocorrencias(alfabeto, target))
    prob_conjunta = calc_probabilidade_conjunta(query, target, alfabeto)
    somatorio = 0

    for X in alfabeto:
        info_mutua = prob_conjunta[X] * np.log2(prob_conjunta[X] / (prob_x[X] * prob_y))
        mask_isnan = np.isnan(info_mutua)
        mask_isinf = np.isinf(info_mutua)
        mask = np.logical_and(np.logical_not(mask_isinf), np.logical_not(mask_isnan))

        info_mutua = info_mutua[mask]
        somatorio += sum(info_mutua)

    return somatorio


'''
FUNÇÃO que calcula a informação mútua de 2 fontes em todas as janelas (definidas através do valor do passo)
'''


def inf_mutua_janelas(passo, num_janelas, query, target, alfabeto):
    informacoes_mutuas = np.zeros(num_janelas)
    for i in range(num_janelas):
        informacoes_mutuas[i] = cria_informacao_cada_janela(query, target[i * passo: (i * passo) + len(query)],
                                                            alfabeto)
    print(f"infoMutua: {informacoes_mutuas}")
    return informacoes_mutuas


'''
FUNÇÃO do exercício 6a)
- Calcula a informação mútua da query e do target, usando o exemplo do pdf
'''


def exercicio_6a():
    query = [2, 6, 4, 10, 5, 9, 5, 8, 0, 8]
    target = [6, 8, 9, 7, 2, 4, 9, 9, 4, 9,
              1, 4, 8, 0, 1, 2, 2, 6, 3, 2,
              0, 7, 4, 9, 5, 4, 8, 5, 2, 7,
              8, 0, 7, 4, 8, 5, 7, 4, 3, 2,
              2, 7, 3, 5, 2, 7, 4, 9, 9, 6]
    alfabeto1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    passo = 1
    print("QUERY: Exemplo --- TARGET: Exemplo")
    num_janelas = cria_num_janelas(query, target, passo)
    inf_mutua_janelas(passo, num_janelas, query, target, alfabeto1)


'''
FUNÇÃO que
- cria um histograma consoante o alfabeto recebido e o nº de ocorrências de cada símbolo do alfabeto
- é dedicada aos exercícios 6b e 6c
'''


def histograma_plotted(alfabeto, ocorrencias):
    plt.plot(alfabeto, ocorrencias, "o")
    plt.show()


'''
FUNÇÃO do exercício 6b)
- calcula a informação mútua de 1 query e 2 targets definidos pelo utilizador
'''


def exercicio_6b(data, alfabeto, filename_target):
    passo = int(0.25 * len(data))

    ocorrencias_target, alfabeto_target, data_target1 = extrair_audio(filename_target)
    num_janelas = cria_num_janelas(data, data_target1, passo)
    print(f"QUERY: saxwriff.wav --- TARGET: \"{filename_target}\"")
    inf_mutua = inf_mutua_janelas(passo, num_janelas, data, data_target1, alfabeto)
    histograma_plotted(np.arange(0, num_janelas), inf_mutua)
    print()


'''
FUNÇÃO do exercício 6c)
- Calcula a informação mútua de 2 fontes em cada janela
- Mostra o seu histograma 
- Calcula a informação mútua máxima
- Apresenta a informação mútua por ordem decrescente
'''


def exercicio_6c(data, alfabeto, filename_target):
    passo = int(0.25 * len(data))
    ocorrencias_target, alfabeto_target, data_target = extrair_audio(filename_target)
    num_janelas = cria_num_janelas(data, data_target, passo)
    print(f"QUERY: saxwriff.wav --- TARGET: \"{filename_target}\"")
    inf_mutua = inf_mutua_janelas(passo, num_janelas, data, data_target, alfabeto)
    histograma_plotted(np.arange(0, num_janelas), inf_mutua)
    print(f"Inf mútua máxima: {np.amax(inf_mutua)}")
    print(f"Inf mútua ordem decrescente: {np.sort(inf_mutua)[::-1]}")
    print()


def main(filename):
    # def main(filename):
    # Condições para verificar extensão do ficheiro, afim de chamar a função correspondente

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
        ocorrencias, alfabeto, data = extrair_audio(filename)
        # Se for um audio stereo usamos apenas o canal esquerdo


    # Condição para textos
    elif filename.lower().endswith('.txt'):
        with open(filename) as f:
            data = np.asarray(list(f.read()))
        data_letras, alfabeto = retirar_texto(data)
        data = cria_array_numeros(data_letras)
        ocorrencias = texto(data, alfabeto)

    # Ex 1
    histograma(alfabeto, ocorrencias)

    # Ex 2 e 3
    print("EX 2 e 3")
    entropia, probabilidade = calc_entropia(ocorrencias)
    print(f"Entropia: {entropia}")
    print("-----------------------------")

    # Ex 4
    print("EX 4")
    media_ponderada, length = calc_media_ponderada(data, probabilidade)
    calc_variancia_ponderada(probabilidade, media_ponderada, length)
    print("-----------------------------")

    # Ex 5
    print("EX 5")
    agrupar_simbolos(data, len(alfabeto))
    print("-----------------------------")

    # Condição para não correr o resto do programa se o ficheiro for uma imagem ou texto
    if filename.lower().endswith('.bmp') or filename.lower().endswith('.txt'):
        exit(1)

    # Ex 6a
    print("EX 6a")
    exercicio_6a()
    print("-----------------------------")

    # Ex 6b
    print("EX 6b")
    exercicio_6b(data, alfabeto, "target01 - repeat.wav")
    exercicio_6b(data, alfabeto, "target02 - repeatNoise.wav")
    print("-----------------------------")

    # Ex 6c
    print("EX 6c")
    exercicio_6c(data, alfabeto, "Song01.wav")
    exercicio_6c(data, alfabeto, "Song02.wav")
    exercicio_6c(data, alfabeto, "Song03.wav")
    exercicio_6c(data, alfabeto, "Song04.wav")
    exercicio_6c(data, alfabeto, "Song05.wav")
    exercicio_6c(data, alfabeto, "Song06.wav")
    exercicio_6c(data, alfabeto, "Song07.wav")


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    while True:
        print("Escolha o ficheiro a ler:\n"
              "1 - \"lena.bmp\"\n"
              "2 - \"binaria.bmp\"\n"
              "3 - \"ct1.bmp\"\n"
              "4 - \"texto.txt\"\n"
              "5 - \"saxriff.wav\"\n"
              "OPÇÃO: ", end="")
        escolha = int(input())

        if escolha == 1:
            print("NOME DO FICHEIRO: \"lena.bmp\"")
            print("-----------------------------")
            main("lena.bmp")
            exit(0)

        elif escolha == 2:
            print("NOME DO FICHEIRO: \"binaria.bmp\"")
            print("-----------------------------")
            main("binaria.bmp")
            exit(0)

        elif escolha == 3:
            print("NOME DO FICHEIRO: \"ct1.bmp\"")
            print("-----------------------------")
            main("ct1.bmp")
            exit(0)

        elif escolha == 4:
            print("NOME DO FICHEIRO: \"texto.txt\"")
            print("-----------------------------")
            main("texto.txt")
            exit(0)

        elif escolha == 5:
            print("NOME DO FICHEIRO: \"saxriff.wav\"")
            print("-----------------------------")
            main("saxriff.wav")
            exit(0)

        else:
            print("Opção errada")
            exit(1)
