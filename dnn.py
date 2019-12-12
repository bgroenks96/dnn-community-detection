import os
import os.path
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

def build_dense(input_dims, output_dims, hidden_dims=64, metrics=['acc'], optimizer='adam',
                loss='categorical_crossentropy', reg=l2(1.0E-5), num_hidden_layers=4):
    input_0 = Input((input_dims,))
    x = input_0
    for i in range(num_hidden_layers):
        x = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(x)
    dense_out = Dense(output_dims, activation='softmax')(x)
    model = Model(inputs=input_0, outputs=dense_out)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def load_run(run, dataset='mnist', num_layers=1, num_epochs=1, base_path=os.getcwd()):
    filename = os.path.join(base_path,
                            'models',
                            dataset,
                            f'run_{run}',
                            f'run_{run}-layers_{num_layers}-epoch_{num_epochs:02d}.hdf5')
    return load_model(filename)
