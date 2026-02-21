"""
Módulo de definição do modelo LSTM adaptado para o Datathon
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_lstm_model(
    sequence_length: int,
    n_features: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    n_layers: int = 2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Cria um modelo LSTM dinâmico para permitir o Grid Search.
    """
    model = Sequential()
    
    # Camada de Entrada
    model.add(Input(shape=(sequence_length, n_features)))

    # Construção dinâmica das camadas LSTM
    for i in range(n_layers):
        # Retorna sequências apenas se não for a última camada LSTM
        return_sequences = (i < n_layers - 1)
        model.add(LSTM(units=lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    # Camada de Saída (1 valor contínuo para 'Defasagem' normalizada)
    model.add(Dense(1, activation='linear'))

    # Compilação
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model