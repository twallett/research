#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, Reshape
)

MLP_2LAYER_HIDDEN = [64]
MLP_4LAYER_HIDDEN = [64, 32, 16]
MLP_8LAYER_HIDDEN = [128, 64, 32, 16, 8, 4]
MLP_DROPOUT = 0.3
MLP_DROPOUT_DEEP = [0.3, 0.3, 0.2, 0.2, 0.1]

CNN_2LAYER_FILTERS = [32]
CNN_4LAYER_FILTERS = [32, 64]
CNN_8LAYER_FILTERS = [32, 64, 128, 64, 32]
CNN_DENSE = [64, 32]
CNN_DROPOUT = 0.3
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

OUTPUT_UNITS = 1
OUTPUT_ACTIVATION = 'sigmoid'
LOSS_FN = 'binary_crossentropy'
METRICS = ['AUC', 'Precision', 'Recall']
OPTIMIZER = 'adam'

class Models:
    
    @staticmethod
    def create_2layer_mlp(input_shape):
        model = Sequential([
            Dense(MLP_2LAYER_HIDDEN[0], activation='relu', input_shape=(input_shape,)),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model

    @staticmethod
    def create_4layer_mlp(input_shape):
        model = Sequential([
            Dense(MLP_4LAYER_HIDDEN[0], activation='relu', input_shape=(input_shape,)),
            Dropout(MLP_DROPOUT),
            Dense(MLP_4LAYER_HIDDEN[1], activation='relu'),
            Dropout(MLP_DROPOUT),
            Dense(MLP_4LAYER_HIDDEN[2], activation='relu'),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model

    @staticmethod
    def create_8layer_mlp(input_shape):
        model = Sequential([
            Dense(MLP_8LAYER_HIDDEN[0], activation='relu', input_shape=(input_shape,)),
            Dropout(MLP_DROPOUT_DEEP[0]),
            Dense(MLP_8LAYER_HIDDEN[1], activation='relu'),
            Dropout(MLP_DROPOUT_DEEP[1]),
            Dense(MLP_8LAYER_HIDDEN[2], activation='relu'),
            Dropout(MLP_DROPOUT_DEEP[2]),
            Dense(MLP_8LAYER_HIDDEN[3], activation='relu'),
            Dropout(MLP_DROPOUT_DEEP[3]),
            Dense(MLP_8LAYER_HIDDEN[4], activation='relu'),
            Dropout(MLP_DROPOUT_DEEP[4]),
            Dense(MLP_8LAYER_HIDDEN[5], activation='relu'),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model

    @staticmethod
    def create_2layer_cnn(input_shape):
        model = Sequential([
            Reshape((input_shape, 1), input_shape=(input_shape,)),
            Conv1D(filters=CNN_2LAYER_FILTERS[0], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model

    @staticmethod
    def create_4layer_cnn(input_shape):
        model = Sequential([
            Reshape((input_shape, 1), input_shape=(input_shape,)),
            Conv1D(filters=CNN_4LAYER_FILTERS[0], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            MaxPooling1D(pool_size=CNN_POOL_SIZE, padding='same'),
            Conv1D(filters=CNN_4LAYER_FILTERS[1], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(CNN_DENSE[1], activation='relu'),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model

    @staticmethod
    def create_8layer_cnn(input_shape):
        model = Sequential([
            Reshape((input_shape, 1), input_shape=(input_shape,)),
            Conv1D(filters=CNN_8LAYER_FILTERS[0], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            MaxPooling1D(pool_size=CNN_POOL_SIZE, padding='same'),
            Conv1D(filters=CNN_8LAYER_FILTERS[1], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            MaxPooling1D(pool_size=CNN_POOL_SIZE, padding='same'),
            Conv1D(filters=CNN_8LAYER_FILTERS[2], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            Conv1D(filters=CNN_8LAYER_FILTERS[3], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            Conv1D(filters=CNN_8LAYER_FILTERS[4], kernel_size=CNN_KERNEL_SIZE, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(CNN_DENSE[0], activation='relu'),
            Dropout(CNN_DROPOUT),
            Dense(CNN_DENSE[1], activation='relu'),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION)
        ])
        model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=METRICS)
        return model