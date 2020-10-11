import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def plot_image(predictions_array, true_label, img, classes):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                         100 * np.max(predictions_array),
                                         classes[true_label]),
               color=color)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, classes = load_dataset()
    plt.figure("one Image")
    plt.imshow(train_x[0])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('This is {}'.format(classes[train_y[0]]))
    print(train_x.shape)
    print(train_y.shape)
    print(train_y)
    train_x, test_x = train_x / 255.0, test_x / 255.0
    train_y = train_y.reshape((train_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))

    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(6))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=10,
                        validation_data=(test_x, test_y))
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    prob_model = tf.keras.Sequential([model,
                                      tf.keras.layers.Softmax()])

    predictions = prob_model.predict(test_x)

    num_rows, num_cols = 5, 5
    plt.figure()
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i+1)
        plot_image(predictions[i], test_y[i], test_x[i], classes)
    plt.tight_layout()

    plt.show()
