import jpeg.encoder, jpeg.decoder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Imagens disponíveis

nome_image = 'baboon.png'
# nome_image = 'dog.jpg'
# nome_image = 'lena.jpg'
# nome_image = 'bob-esponja.jpg'

path_orig = 'Images/' # Caminho em que a imagem original está

# Leitura da Imagem
img = Image.open(path_orig + nome_image) # Abre a imagem
src_img = np.asarray(img).astype(np.float64) # Conversão da imagem

plt.figure('Imagens') # Cria a figura para plotar as imagens
plt.subplot(1,2,1) # Seleciona onde plotar
plt.imshow(img) # Plota a imagem original

img_larg, img_alt = img.size # pega a largura e a altura da imagem

# Codificação
codificador = jpeg.encoder.Encoder(img) # Cria uma instância da classe do codificador
comprimido = codificador.codificar() # Realiza o processo de codificação JPEG
header = comprimido['header'] # Criação do cabeçalho em que estarão algumas informações necessárias para a descompressão

# Decodificação
decodificador = jpeg.decoder.Decoder(src_img, header, comprimido,(codificador.larg, codificador.alt))
# Cria uma instância da classe do decodificador
decoded_img = decodificador.decodificar() # Realiza o processo de decodificação JPEG

img = np.asarray(decoded_img).copy()
img_final = img[:img_alt, :img_larg, :]

# Calcula e printa o erro quadrático médio
print(f'Erro quadrático médio: {np.mean((src_img - img_final.astype(np.float64))**2)}') 

img_final = Image.fromarray(img_final) # Transforma em imagem

plt.subplot(1,2,2) # Seleciona onde plotar
plt.imshow(img_final) # Plota a imagem final após a compressção e descompressão
plt.show() # Realiza o plot das duas imagens configuradas anteriormente