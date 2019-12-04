from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, X_train.shape[1] * X_train.shape[2]))
X_train = (X_train > 0).astype(np.float32)
X_test = X_test.reshape((-1, X_test.shape[1] * X_test.shape[2]))
X_test = (X_test > 0).astype(np.float32)
train_labels = y_train
test_labels = y_test
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def subsample(n, labels):
    return np.concatenate([np.random.choice(np.where(labels == c)[0], n // 10) for c in range(10)], axis=0)

def subsample_train(n):
    return subsample(n, train_labels)

def subsample_test(n):
    return subsample(n, test_labels)

def subsample_all(n):
    return subsample(n, np.concatenate(train_labels, test_labels))
