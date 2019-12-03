from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

def build_dense(input_dims, hidden_dims=64, metrics=['acc'], optimizer='adam',
                loss='categorical_crossentropy', reg=l2(1.0E-5)):
    input_0 = Input((input_dims,))
    dense_1 = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(input_0)
    dense_2 = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(dense_1)
    dense_3 = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(dense_2)
    dense_4 = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(dense_3)
    dense_out = Dense(10, activation='softmax')(dense_4)
    model = Model(inputs=input_0, outputs=dense_out)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
