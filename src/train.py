"""
Script principal de treinamento com Grid Search para o Datathon
"""
import os
import json
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Importe sua classe de pre-processamento e a função do modelo
from src.preprocessing import DataPreprocessor 
from src.models.lstm import create_lstm_model
from src.config import FEATURES, MODELS_DIR, SCALERS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÃO DO GRID SEARCH
# ============================================================
GRID_SEARCH_PARAMS = {
    "lstm_units": [32, 64],
    "dropout_rate": [0.1, 0.2],
    "batch_size": [16, 32],
    "epochs": [50], # Mantenha baixo para testes rápidos
    "learning_rate": [0.001],
    "n_layers": [1, 2]
}

def run_grid_search(df, preprocessor):
    logger.info("="*50)
    logger.info("INICIANDO GRID SEARCH")
    logger.info("="*50)

    # 1. Preparar os dados usando seu preprocessor
    processed_df = preprocessor.filter_features(df)
    processed_df['Fase'] = processed_df['Fase'].astype('str')
    cleaned_df = processed_df.dropna()
    cleaned_df = preprocessor.clean_data(cleaned_df)
    cleaned_df = preprocessor.encode_categorical(cleaned_df, is_training=True)

    # Normalizar
    preprocessor.fit_scalers(cleaned_df)
    features_scaled, target_scaled = preprocessor.transform(cleaned_df)
    
    # Criar sequências
    X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    n_features = X.shape[2]
    seq_length = X.shape[1]

    # Salvar preprocessor
    os.makedirs(SCALERS_DIR, exist_ok=True)
    preprocessor.save_preprocessor(SCALERS_DIR / "datathon_preprocessor.joblib")

    # 2. Gerar combinações
    param_keys = list(GRID_SEARCH_PARAMS.keys())
    all_combinations = list(product(*GRID_SEARCH_PARAMS.values()))
    
    best_loss = float('inf')
    best_params = None
    best_model = None

    # KFold simplificado (se não for série temporal estrita, KFold comum é melhor)
    kf = KFold(n_splits=3, shuffle=False)

    logger.info(f"Testando {len(all_combinations)} combinações...")

    for idx, combo in enumerate(all_combinations, 1):
        params = dict(zip(param_keys, combo))
        logger.info(f"\n[{idx}/{len(all_combinations)}] Testando hiperparâmetros: {params}")

        fold_losses = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = create_lstm_model(
                sequence_length=seq_length,
                n_features=n_features,
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                n_layers=params['n_layers'],
                learning_rate=params['learning_rate']
            )

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop],
                verbose=0 # Mude para 1 se quiser ver o progresso época por época
            )

            min_val_loss = min(history.history['val_loss'])
            fold_losses.append(min_val_loss)
            
            # Limpar memória do Keras
            tf.keras.backend.clear_session()

        mean_val_loss = np.mean(fold_losses)
        logger.info(f"Loss Médio de Validação (MSE): {mean_val_loss:.5f}")

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_params = params.copy()
            logger.info("🌟 Novo melhor modelo encontrado!")

    # 3. Treinar o modelo final com os melhores parâmetros
    logger.info("="*50)
    logger.info(f"Treinando modelo FINAL com os melhores params: {best_params}")
    
    final_model = create_lstm_model(
        sequence_length=seq_length,
        n_features=n_features,
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        n_layers=best_params['n_layers'],
        learning_rate=best_params['learning_rate']
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(MODELS_DIR / "datathon_best_lstm.keras"),
        monitor='val_loss',
        save_best_only=True
    )

    # Treinando com todos os dados divididos simples (80/20) para o checkpoint
    split_idx = int(len(X) * 0.8)
    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
    y_train_final, y_val_final = y[:split_idx], y[split_idx:]

    final_model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=best_params['epochs'] * 2, # Dá mais tempo pro melhor modelo convergir
        batch_size=best_params['batch_size'],
        callbacks=[checkpoint, EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1
    )

    # Salvar log dos parâmetros
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info("Treinamento finalizado. Modelo e Scalers salvos!")

def main():
    caminho_script = Path(__file__).resolve().parent
    raiz_projeto = caminho_script.parents[0]
    caminho_arquivo = raiz_projeto / 'data' / 'datathon_pede_2024.xlsx'
    
    valores_invalidos = ["#DIV/0!", "INCLUIR", "#N/A"]
    df = pd.read_excel(caminho_arquivo, sheet_name='PEDE2024', na_values=valores_invalidos)

    preprocessor = DataPreprocessor(validate_data=True, scaler_type="minmax")
    
    run_grid_search(df, preprocessor)

if __name__ == "__main__":
    main()