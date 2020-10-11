import tensorflow as tf
from net05_ResNeXt import resnext

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = resnext.ResNeXt50()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=128,
          epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)
model.summary()
