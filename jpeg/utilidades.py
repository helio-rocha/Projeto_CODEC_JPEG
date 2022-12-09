import numpy as np

# Tabelas de quantização
Q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]]) # Tabela de quantização de luminância

Q_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]]) # Tabela de quantização de crominância

zigzag_order = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,
                         40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,
                         43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,
                         46,53,60,61,54,47,55,62,63])
# Tabela da ordem em que será realizado o zig-zag na codificação

# Funções gerais
def reconstruct_from_blocks(blocos, img_larg):
    reconstruido = [] # Inicializa o vetor que armazena os blocos concatenados
    N_blocos = int(img_larg / 8) # Calcula o número de blocos na imagem

    for n in range(0, len(blocos) - N_blocos + 1, N_blocos):
        res = np.concatenate(blocos[n : n + N_blocos], axis=1) # Concatena os blocos
        reconstruido.append(res) # Adiciona os blocos concatenados na lista

    return np.concatenate(reconstruido) # Retorna a imagem reconstruida a partir dos blocos

def transform_to_block(image): # Converte a imagem em vários blocos 8x8
    img_larg, img_alt = image.shape # Pega a largura e altura da imagem
    blocos = [] # Cria a matriz dos blocos
    for i in range(0, img_larg, 8): # Varredura com incremento de 8 na largura
        for j in range(0, img_alt, 8): # Varredura com incremento de 8 na altura
            blocos.append(image[i:i+8,j:j+8]) # Adiciona um bloco 8x8 ao vetor de blocos
    return blocos # Retorna o vetor com os blocos

def zero_padding(matriz): # Completa a matriz com zeros
    ncolunas, nlinhas = matriz.shape[0], matriz.shape[1] # Pega o número de linhas e colunas

    if (ncolunas % 8 != 0): # Se o número de colunas não for múltiplo de 8 completa com zeros
        img_larg = ncolunas // 8 * 8 + 8
    else: # Caso contrário, não altera a matriz
        img_larg = ncolunas

    if (nlinhas % 8 != 0): # Se o número de linhas não for múltiplo de 8 completa com zeros
        img_alt = nlinhas // 8 * 8 + 8
    else: # Caso contrário, não altera a matriz
        img_alt = nlinhas

    # Copia dos dados para a nova matriz
    nova_matriz = np.zeros((img_larg, img_alt), dtype=np.float64) # Matríz com zeros
    for y in range(ncolunas):
        for x in range(nlinhas):
            nova_matriz[y][x] = matriz[y][x] # Sobrepõe os zeros com os dados originais, quando presentes
    return nova_matriz # Retorna a matriz completada com zeros

