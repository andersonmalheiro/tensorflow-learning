from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Carregando base de dados do IMDB
imdb = keras.datasets.imdb

# Dividindo base entre treino e testes
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# Função para decodificar o texto das avaliações
# A base por padrão é composta de inteiros, essa função obtem as palavras correspondentes no dicionario
# Os primeiros indices são reservados
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # desconhecido
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Padronizando os dados de treino e teste em tensores de 256 posições
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=256)


# contagem de vocabulário usada para as resenhas de filmes (10.000 palavras)
vocab_size = 10000

# Construindo o modelo
model = keras.Sequential()

# Camada de incorporação.
# Essa camada pega o vocabulário codificado por inteiros e procura o vetor de incorporação para cada palavra-índice.
# Esses vetores são aprendidos com o treinamento do modelo.
# Os vetores adicionam uma dimensão ao array de saída. As dimensões resultantes são: (lote, sequência, incorporação)
model.add(keras.layers.Embedding(vocab_size, 16))


# Camada para retornar um vetor de saída de tamanho fixo para cada exemplo,
# calculando a média da dimensão da seqüência.
# Isso permite que o modelo manipule a entrada de comprimento variável, da maneira mais simples possível.
model.add(keras.layers.GlobalAveragePooling1D())


# Esse vetor de saída de comprimento fixo é canalizado
# através de uma camada (Densa) totalmente conectada com 16 unidades ocultas.
model.add(keras.layers.Dense(16, activation=tf.nn.relu))


# A última camada é densamente conectada com um único nó de saída.
# Usando a função de ativação sigmoid, esse valor é um float entre 0 e 1,
# representando uma probabilidade ou nível de confiança.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


# Configuração do modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# Conjunto de dados para validação
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# Treinando modelo
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# Avaliando modelo
results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
