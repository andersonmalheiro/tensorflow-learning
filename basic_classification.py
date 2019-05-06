from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Carrega a base de dados de imagens
fashion_mnist = keras.datasets.fashion_mnist

# Divide os dados em base de treinamento e testes
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Construindo rede neural
model = keras.Sequential([
    # Camada para converter as imagens em vetores 1D de 28x28=784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),

    # Camada com 128 neurônios
    keras.layers.Dense(128, activation=tf.nn.relu),

    # Camada com 10 nós de saída que calculam a probabilidade
    # de uma imagem pertencer a uma das categorias
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compila o modelo, passando função de custo, otimizado e métricas
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Passo de treinamento
model.fit(train_images, train_labels, epochs=5)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

# Predições
predictions = model.predict(test_images)

print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Criar o gráfico das primeiras X imagens, o label previsto e o label correto
# Colorir as previsões corretas a azul e as incorretas a vermelho
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, (2 * num_cols), (2 * i + 2))
    plot_value_array(i, predictions, test_labels)

plt.show()