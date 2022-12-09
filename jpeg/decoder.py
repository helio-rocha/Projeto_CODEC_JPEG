from PIL import Image
from scipy.fftpack import idct
import numpy as np
import jpeg.utilidades as utilidades
import cv2
from cv2 import normalize
from .huffman import H_Decoder, DC, AC, LUMINANCE, CHROMINANCE

class Decoder(): # Classe do codificador
    def __init__(self, image, header, compressed, img_info): # Construtor da classe
        self.larg  = img_info[0] # Largura da imagem
        self.alt = img_info[1] # Altura da imagem
        self.bits   = compressed['data'] # Bits da imagem compactada
        self.remaining_bits_length = header['remaining_bits_length'] # Bits remanescentes
        self.dsls   = header['data_slice_lengths']
        self.image  = image # Imagem

    def idct(self, blocos): # Realiza a transformada discreta (IDCT) do cosseno inversa nos blocos da imagem
        return idct(idct(blocos, axis=0, norm = 'ortho'), axis=1, norm = 'ortho')

    def dequantization(self, G, type): # Realiza a quantização inversa
        if (type == 'l'): # Utiliza a tabela da luminância
            return(np.multiply(G, utilidades.Q_y))
        elif (type == 'c'): # Utiliza a tabela da crominância
            return(np.multiply(G, utilidades.Q_c))

    def upsampling(self, cb, cr, n_linhas, n_col):
        up_cb = cv2.resize(cb, dsize=(n_col, n_linhas)) # Realiza o UpSampling da Crominância Azul
        up_cr = cv2.resize(cr, dsize=(n_col, n_linhas)) # Realiza o UpSampling da Crominância Vermelha

        return (up_cb, up_cr) # Retorna os valores após o UpSampling

    def decodificar(self):

        bits = self.bits.to01() # lê o array de bits codificado
        dsls = self.dsls  # data_slice_lengths

        sliced = {
            LUMINANCE: {
                DC: bits[:dsls[0]],
                AC: bits[dsls[0]:dsls[0] + dsls[1]]
            },
            CHROMINANCE: {
                DC: bits[dsls[0] + dsls[1]:dsls[0] + dsls[1] + dsls[2]],
                AC: bits[dsls[0] + dsls[1] + dsls[2]:]
            }
        }
        cb, cr = np.split(H_Decoder(sliced[CHROMINANCE], CHROMINANCE).decode(), 2) # Realiza a decodificação Huffman e encontra as crominâncias
        y = H_Decoder(sliced[LUMINANCE], LUMINANCE).decode() # Realiza a decodificação Huffman e encontra luminância

        # Dequantização
        dqnt_Y  = self.dequantization(y, 'l') # Dequantização usando a matriz de luminância
        dqnt_Cb = self.dequantization(cb, 'c') # Dequantização usando a matriz de Crominância
        dqnt_Cr = self.dequantization(cr, 'c') # Dequantização usando a matriz de Crominância

        # Calcula a DCT inversa
        idct_Y  = self.idct(dqnt_Y)
        idct_Cb = self.idct(dqnt_Cb)
        idct_Cr = self.idct(dqnt_Cr)

        # Reconstroi a imagem a partir dos blocos
        Y  = utilidades.reconstruct_from_blocks(idct_Y, self.larg)
        Cb = utilidades.reconstruct_from_blocks(idct_Cb, self.larg)
        Cr = utilidades.reconstruct_from_blocks(idct_Cr, self.larg)

        # Faz o UpSampling da crominância azul e vermelha
        Cb, Cr = self.upsampling(Cb, Cr, self.alt, self.larg)

        # Desfaz o shift de 128 feito na codificação
        img = np.dstack((Y, Cb, Cr)) + 128.0

        img = normalize(img, 0, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Normaliza os valores
        img = Image.fromarray(img, 'YCbCr').convert('RGB') # Converte para RGB

        return img # Retorna a imagem decodificada