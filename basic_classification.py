from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_image, plot_value_array

# Carrega a base de dados de imagens
fashion_mnist = keras.datasets.fashion_mnist

# Divide os dados em base de treinamento e testes
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

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
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Passo de treinamento
model.fit(train_images, train_labels, epochs=10)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

# Predições
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, class_names, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()
