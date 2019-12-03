from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_and_preprocess():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((-1, X_train.shape[1] * X_train.shape[2]))
    X_train = (X_train > 0).astype(np.float32)
    X_test = X_test.reshape((-1, X_test.shape[1] * X_test.shape[2]))
    X_test = (X_test > 0).astype(np.float32)
    y_train_orig, y_test_orig = y_train, y_test
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (X_train, y_train), (X_test, y_test)
