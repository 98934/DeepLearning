import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import decode_predictions
from tensorflow.keras.applications.resnet import preprocess_input


import os
import matplotlib.pyplot as plt


cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

resnet50 = ResNet50(include_top=True, weights=None, input_tensor=None,
                    input_shape=(32, 32, 3), pooling=None, classes=10)

resnet50.summary()
resnet50.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['sparse_categorical_accuracy'])


checkpoint_save_path = './checkpoint/resnet50.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------load the model------------')
    resnet50.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = resnet50.fit(x_train, y_train, batch_size=128,
                       epochs=5, validation_data=(x_test, y_test),
                       validation_freq=1, callbacks=[cp_callback])

resnet50.summary()

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
