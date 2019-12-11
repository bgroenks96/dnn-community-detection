from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

def build_dense(input_dims, hidden_dims=64, metrics=['acc'], optimizer='adam',
                loss='categorical_crossentropy', reg=l2(1.0E-5), num_hidden_layers=4):
    input_0 = Input((input_dims,))
    x = input_0
    for i in range(num_hidden_layers):
        x = Dense(hidden_dims, activation='relu', kernel_regularizer=reg)(x)
    dense_out = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_0, outputs=dense_out)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
