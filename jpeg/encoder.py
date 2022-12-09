from PIL import Image
from scipy.fftpack import dct
import numpy as np
import jpeg.utilidades as utilidades
from bitarray import bitarray, bits2bytes
from .huffman import H_Encoder, DC, AC, LUMINANCE, CHROMINANCE

class Encoder(): # Classe do codificador
    def __init__(self, image): # Construtor da classe
        self.image  = image # Imagem
        self.larg  = None # Largura da imagem
        self.alt = None # Altura da imagem

    def dct(self, blocos): # Realiza a transformada discreta (DCT) do cosseno nos blocos da imagem
        return dct(dct(blocos, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def quantizar(self, G, type): # Realiza a quantização, ou seja, a remoção dos dados de alta frequência
        if (type == 'l'): # Utiliza a tabela de luminância
            return(np.divide(G, utilidades.Q_y).round().astype(np.float64))
        elif (type == 'c'): # Utiliza a tabela de crominância
            return(np.divide(G, utilidades.Q_c).round().astype(np.float64))

    def downsampling(self, matrix, k=2, type=2):
        if type == 1: # Realiza o DownSampling somente nas colunas
            ds_img = matrix[:,0::k]
        elif type == 2: # Realiza o DownSampling nas colunas e nas linhas
            ds_img = matrix[0::k,0::k]
        else: # Não Realiza DownSampling
            ds_img = matrix

        return ds_img # Retorna a matriz após a realização do DownSampling

    def codificar(self):
        # Largura e altura da imagem
        alt_src_img, larg_src_img = self.image.size # pega a altura e a largura da imagem
        print(f'Image: Altura = {alt_src_img}, Largura = {larg_src_img}') # Printa a altura e a largura

        # Converte para matriz Numpy
        matriz_src_img = np.asarray(self.image)

        # Converte 'RGB' para 'YCbCr'
        img_ycbcr = Image.fromarray(matriz_src_img).convert('YCbCr')
        img_ycbcr = np.asarray(img_ycbcr).astype(np.float64)

        # Subtrai 128 para fazer o shifted block
        Y   = img_ycbcr[:,:,0] - 128
        Cb  = img_ycbcr[:,:,1] - 128
        Cr  = img_ycbcr[:,:,2] - 128

        # Aplica o DownSampling no Cb e Cr
        Cb = self.downsampling(Cb, alt_src_img, larg_src_img)
        Cr = self.downsampling(Cr, alt_src_img, larg_src_img)

        # Preenche com zeros se for necessário
        Y  = utilidades.zero_padding(Y)
        Cb = utilidades.zero_padding(Cb)
        Cr = utilidades.zero_padding(Cr)

        # Salva o novo tamanho
        self.alt, self.larg = Y.shape

        # Transforma a imagem em vários blocos
        blocos_Y  = utilidades.transform_to_block(Y)
        blocos_Cb = utilidades.transform_to_block(Cb)
        blocos_Cr = utilidades.transform_to_block(Cr)

        # Calcula a DCT
        Y_dct  = self.dct(blocos_Y)
        Cb_dct = self.dct(blocos_Cb)
        Cr_dct = self.dct(blocos_Cr)

        # Realiza a quantização
        Y_qnt  = self.quantizar(Y_dct, 'l') # Quantização usando a matriz de luminância
        Cb_qnt = self.quantizar(Cb_dct, 'c') # Quantização usando a matriz de crominância
        Cr_qnt = self.quantizar(Cr_dct, 'c') # Quantização usando a matriz de crominância

        # Codificação Huffman
        codificado = {
            LUMINANCE: H_Encoder(Y_qnt, LUMINANCE).encode(), # Codificação da luminância
            CHROMINANCE: H_Encoder(
                np.vstack((Cb_qnt, Cr_qnt)),
                CHROMINANCE
            ).encode() # Codificação das crominâncias
        }

        # Reordena a luminância e a crominância
        order = (codificado[LUMINANCE][DC], codificado[LUMINANCE][AC],
                 codificado[CHROMINANCE][DC], codificado[CHROMINANCE][AC])

        bits = bitarray(''.join(order)) # Cria um array de bits, que são a imagem comprimida

        return {
            'data': bits, # bits comprimidos
            'header': { # Metadados
                'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits), # bits remanescentes
                'data_slice_lengths': tuple(len(d) for d in order)
        } # Retorna os bits comprimidos juntamente com alguns metadados que ajudarão na decodificação da imagem
    }