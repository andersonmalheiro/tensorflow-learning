import tensorflow as tf
from tensorflow.keras import layers

# Contruindo modelo
model = tf.keras.Sequential([
  # Adiciona duas camadas conectadas densamente com 64 unidades
  layers.Dense(64, activation='relu'),
  layers.Dense(64, activation='relu'),

  # Adiciona uma camada softmax com 10 unidades de saida
  layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])