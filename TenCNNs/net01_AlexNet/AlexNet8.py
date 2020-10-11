import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os

np.set_printoptions(threshold=np.inf)
# print(np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_AlexNet8():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=96, kernel_size=(3, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='AlexNet8')

    return model


alex_net = create_AlexNet8()
alex_net.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['sparse_categorical_accuracy'])
# alex_net.summary()

checkpoint_save_path = './checkpoint/AlexNet8.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------load the model------------')
    alex_net.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = alex_net.fit(x_train, y_train, batch_size=128,
                       epochs=5, validation_data=(x_test, y_test),
                       validation_freq=1, callbacks=[cp_callback])

alex_net.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure('Training Result')
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')

plt.legend()
plt.show()
