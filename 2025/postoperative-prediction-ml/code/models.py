#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from keras_tuner import HyperModel

# Common metrics and loss for all models
METRICS = ['AUC', 'Precision', 'Recall']
LOSS_FN = 'binary_crossentropy'

class TwoLayerMLPHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        units = hp.Int('units', min_value=8, max_value=128, step=32)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Dense(units, activation='relu', input_shape=(self.input_shape,)),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model

class FourLayerMLPHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
        units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
        units_3 = hp.Int('units_3', min_value=8, max_value=16, step=8)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Dense(units_1, activation='relu', input_shape=(self.input_shape,)),
            Dropout(dropout_rate),
            Dense(units_2, activation='relu'),
            Dropout(dropout_rate),
            Dense(units_3, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model

class EightLayerMLPHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        units_1 = hp.Int('units_1', min_value=64, max_value=256, step=64)
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
        units_3 = hp.Int('units_3', min_value=16, max_value=64, step=16)
        units_4 = hp.Int('units_4', min_value=8, max_value=32, step=8)
        units_5 = hp.Int('units_5', min_value=4, max_value=16, step=4)
        units_6 = hp.Int('units_6', min_value=2, max_value=8, step=2)
        dropout_rate_1 = hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)
        dropout_rate_2 = hp.Float('dropout_rate_2', min_value=0.1, max_value=0.4, step=0.1)
        dropout_rate_3 = hp.Float('dropout_rate_3', min_value=0.1, max_value=0.3, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Dense(units_1, activation='relu', input_shape=(self.input_shape,)),
            Dropout(dropout_rate_1),
            Dense(units_2, activation='relu'),
            Dropout(dropout_rate_1),
            Dense(units_3, activation='relu'),
            Dropout(dropout_rate_2),
            Dense(units_4, activation='relu'),
            Dropout(dropout_rate_2),
            Dense(units_5, activation='relu'),
            Dropout(dropout_rate_3),
            Dense(units_6, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model

class TwoLayerCNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        filters = hp.Int('filters', min_value=16, max_value=64, step=16)
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model

class FourLayerCNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16)
        filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        pool_size = hp.Choice('pool_size', values=[2, 3])
        dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
            Conv1D(filters=filters_1, kernel_size=kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=pool_size, padding='same'),
            Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(dense_units, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model

class EightLayerCNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        # Tunable parameters
        filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16)
        filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32)
        filters_3 = hp.Int('filters_3', min_value=64, max_value=256, step=64)
        filters_4 = hp.Int('filters_4', min_value=32, max_value=128, step=32)
        filters_5 = hp.Int('filters_5', min_value=16, max_value=64, step=16)
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5])
        pool_size = hp.Choice('pool_size', values=[2, 3])
        dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=128, step=32)
        dense_units_2 = hp.Int('dense_units_2', min_value=16, max_value=64, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
        
        model = Sequential([
            Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
            Conv1D(filters=filters_1, kernel_size=kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=pool_size, padding='same'),
            Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=pool_size, padding='same'),
            Conv1D(filters=filters_3, kernel_size=kernel_size, activation='relu', padding='same'),
            Conv1D(filters=filters_4, kernel_size=kernel_size, activation='relu', padding='same'),
            Conv1D(filters=filters_5, kernel_size=kernel_size, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(dense_units_1, activation='relu'),
            Dropout(dropout_rate),
            Dense(dense_units_2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=METRICS)
        return model