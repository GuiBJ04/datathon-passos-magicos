import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, Optional, List, Dict
import joblib
import logging
from src.config import SEQUENCE_LENGTH, TRAIN_TEST_SPLIT, FEATURES, TARGET, SCALERS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pre-processamento de dados de acoes para LSTM."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        features: List[str] = FEATURES,
        target: str = TARGET,
        scaler_type: str = "minmax",  # 'minmax' ou 'robust'
        validate_data: bool = True
    ):
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.validate_data = validate_data
        # Escolher tipo de scaler
        if scaler_type == "robust":
            # RobustScaler e menos sensivel a outliers
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        self.scaler_type = scaler_type
        self.is_fitted = False
        self.data_quality_report = None
        self.cleaning_stats: Dict = {}

    def data_show(self, df: pd.DataFrame, n_rows: int = 5):
        # 1. Dimensões e Index
        print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")

        # 2. Amostra dos dados
        print(f"Primeiras {n_rows} linhas:")
        print(df.head(n_rows))
        
        print(f"\nÚltimas {n_rows} linhas:")
        print(df.tail(n_rows))

        # 3. Verificação de Nulos e Tipos
        print("Colunas e Nulos:")
        info_tab = pd.DataFrame({
            'Tipo': df.dtypes,
            'Nulos (Qtd)': df.isnull().sum(),
            'Nulos (%)': (df.isnull().sum() / len(df) * 100).round(2)
        })
        # Mostra apenas colunas que têm nulos ou todas se quiser ver os tipos
        print(info_tab)

        # 4. Estatísticas básicas das Features selecionadas (se existirem no df)
        cols_to_check = [c for c in self.features if c in df.columns]
        if cols_to_check:
            print("Features Selecionadas:")
            print(df[cols_to_check].describe().T[['mean', 'min', '50%', 'max', 'std']])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Realiza limpezas específicas nos dados antes do processamento.
            - Converte 'ALFA' para 0 e mantém apenas o número da 'Fase'.
            - Padroniza colunas de texto para minúsculas.
            """
            # Criar uma cópia para evitar warnings do pandas (SettingWithCopy)
            df_cleaned = df.copy()
            
            # 1. Tratar a coluna 'Fase'
            if 'Fase' in df_cleaned.columns:
                # Garante que seja string, coloca tudo em maiúsculo (para evitar falhas com 'Alfa') e troca por '0'
                df_cleaned['Fase'] = df_cleaned['Fase'].astype(str).str.upper().replace('ALFA', '0')
                # Pega apenas o primeiro caractere (o número) e converte para int
                df_cleaned['Fase'] = df_cleaned['Fase'].str[0].astype(int)
                logger.info("Coluna 'Fase' limpa, tratada e convertida para numérico (int).")
            
            return df_cleaned
        
    def fit_scalers(self, df: pd.DataFrame) -> None:
            # Selecionar apenas colunas numéricas disponíveis
            available_features = [f for f in self.features if f in df.columns]
            df_numeric = df[available_features].select_dtypes(include=[np.number])
            
            print(f"Features numéricas selecionadas para scaler: {df_numeric.columns.tolist()}")
            
            self.numeric_features = df_numeric.columns.tolist() # Salvar para usar no transform
            
            # Fit do scaler de features apenas nos números
            self.feature_scaler.fit(df_numeric.values)
            self.target_scaler.fit(df[[self.target]].values)

            self.is_fitted = True
            
            # CORREÇÃO AQUI: Salvar a lista de nomes das colunas numéricas, não o DataFrame
            self.fitted_features = self.numeric_features 
            
            logger.info(f"Scalers ajustados com {len(self.numeric_features)} features numéricas")
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforma os dados usando os scalers.

        Args:
            df: DataFrame com dados

        Returns:
            Tuple com features e target normalizados
        """
        if not self.is_fitted:
            raise ValueError("Scalers nao ajustados. Execute fit_scalers primeiro.")

        features_scaled = self.feature_scaler.transform(
            df[self.fitted_features].values
        )
        target_scaled = self.target_scaler.transform(
            df[[self.target]].values
        )

        return features_scaled, target_scaled
        
    def filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna o DataFrame contendo apenas as colunas definidas em self.features.
        """
        # Verifica quais colunas da lista de features NÃO estão no Excel
        missing_cols = [col for col in self.features if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"❌ Erro: As seguintes colunas não foram encontradas no Excel: {missing_cols}")
        
        # Se tudo estiver ok, filtra e retorna (mantendo o índice original)
        print(f"✅ Filtrando {len(self.features)} features: {self.features}")
        return df[self.features].copy() # .copy() é importante para evitar avisos de SettingWithCopy
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica colunas do tipo string/object, aplica One-Hot Encoding 
        e atualiza a lista de features da classe.
        """
        # Identificar colunas categóricas (strings ou objects)
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            
        for col in cat_cols:
            df[col] = df[col].astype(str).str.lower()
            
        logger.info(f"Colunas padronizadas para letras minúsculas: {cat_cols}")
        
        # Garantir que a coluna target não seja codificada acidentalmente nas features
        cat_cols = [col for col in cat_cols if col != self.target]

        if not cat_cols:
            logger.info("Nenhuma coluna categórica encontrada para One-Hot Encoding.")
            return df
        
        logger.info(f"Aplicando One-Hot Encoding nas colunas: {cat_cols}")
        
        # Aplicar One-Hot Encoding (dtype=int garante 0 e 1 no lugar de True e False)
        df_encoded = pd.get_dummies(df, columns=cat_cols, dtype=int)
        
        # Atualizar a lista de features da classe com as novas colunas geradas
        # (Mantendo apenas o que não é o target)
        self.features = [col for col in df_encoded.columns if col != self.target]
        
        return df_encoded

def main():
    """Exemplo de uso do preprocessador com validacao."""
    caminho_script = Path(__file__).resolve().parent
    raiz_projeto = caminho_script.parents[1]
    caminho_arquivo = raiz_projeto / 'Datathon' / 'data' / 'datathon_pede_2024.xlsx'
    
    valores_invalidos = ["#DIV/0!", "INCLUIR", "#N/A"]
    
    df = pd.read_excel(
        caminho_arquivo, 
        sheet_name='PEDE2024',
        na_values=valores_invalidos
    )

    print(f"\nDados coletados: {len(df)} registros")

    # Preprocessar COM validacao
    preprocessor = DataPreprocessor(
        features=FEATURES,
        validate_data=True,
        scaler_type="minmax"
    )
    
    processed_df = preprocessor.filter_features(df)
    
    processed_df['Fase'] = processed_df['Fase'].astype('str')
    
    cleaned_df = processed_df.dropna()

    # print(preprocessor.data_show(cleaned_df))
    cleaned_df = preprocessor.clean_data(cleaned_df)

    cleaned_df = preprocessor.encode_categorical(cleaned_df)

    # # Fit e transform
    preprocessor.fit_scalers(cleaned_df)
    features_scaled, target_scaled = preprocessor.transform(cleaned_df)
    
    df_feature_scaled = pd.DataFrame(features_scaled, columns=preprocessor.numeric_features)
    df_target_scaled = pd.DataFrame(target_scaled, columns=[preprocessor.target])

    print(preprocessor.data_show(df_feature_scaled))
    print(preprocessor.data_show(df_target_scaled))



if __name__ == "__main__":
    main()
