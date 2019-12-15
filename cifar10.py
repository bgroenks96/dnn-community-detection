from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = tf.image.rgb_to_hsv(X_train.astype(np.float32) / 255.) \
                  .numpy().reshape((-1, np.prod(X_train.shape[1:])))
X_test = tf.image.rgb_to_hsv(X_test.astype(np.float32) / 255.) \
                  .numpy().reshape((-1, np.prod(X_train.shape[1:])))
train_label_ids = y_train.flatten()
test_label_ids = y_test.flatten()
train_labels = np.array(list(map(lambda i: labels[i], y_train.flatten())))
test_labels = np.array(list(map(lambda i: labels[i], y_test.flatten())))
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def subsample(n, labels):
    return np.concatenate([np.random.choice(np.where(labels == c)[0], n // 10, replace=False) for c in range(10)], axis=0)

def subsample_train(n):
    return subsample(n, train_label_ids)

def subsample_test(n):
    return subsample(n, test_label_ids)

def subsample_all(n):
    return subsample(n, np.concatenate(train_labels, test_labels))
