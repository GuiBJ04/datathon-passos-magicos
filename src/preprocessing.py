from importlib.resources import path
import os
from platform import processor
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from typing import Tuple, Optional, List, Dict, Union
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
        
        self.encoder = OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')
        self.cat_cols = []
        self.encoder_is_fitted = False
        
    def encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding usando Scikit-Learn para garantir consistência
        entre treino e teste.
        """
        df_encoded = df.copy()
        
        if is_training:
            # Identifica colunas categóricas apenas no treino
            self.cat_cols = df_encoded.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
            self.cat_cols = [col for col in self.cat_cols if col != self.target]

            if not self.cat_cols:
                logger.info("Nenhuma coluna categórica encontrada para One-Hot Encoding.")
                return df_encoded

            logger.info(f"Ajustando One-Hot Encoding nas colunas: {self.cat_cols}")
            
            # Padroniza para minúsculas
            for col in self.cat_cols:
                df_encoded[col] = df_encoded[col].astype(str).str.lower()
            
            # FIT e TRANSFORM no treino
            encoded_array = self.encoder.fit_transform(df_encoded[self.cat_cols])
            self.encoder_is_fitted = True
            
        else:
            # Fluxo para dados de TESTE ou PRODUÇÃO
            if not self.encoder_is_fitted:
                raise ValueError("Encoder não ajustado. Execute no modo is_training=True primeiro.")
            
            # Padroniza para minúsculas
            for col in self.cat_cols:
                df_encoded[col] = df_encoded[col].astype(str).str.lower()
                
            # Apenas TRANSFORM no teste
            encoded_array = self.encoder.transform(df_encoded[self.cat_cols])

        # Recupera os nomes das novas colunas geradas pelo encoder
        encoded_col_names = self.encoder.get_feature_names_out(self.cat_cols)
        
        # Cria um DataFrame com as novas colunas
        df_cats = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df_encoded.index)
        
        # Remove as colunas categóricas originais e concatena as novas
        df_encoded = df_encoded.drop(columns=self.cat_cols)
        df_encoded = pd.concat([df_encoded, df_cats], axis=1)
        
        # Atualiza a lista de features da classe
        self.features = [col for col in df_encoded.columns if col != self.target]
        
        return df_encoded

    def data_show(self, data: Union[pd.DataFrame, np.ndarray], n_rows: int = 5, name: str = "Dados"):
        """
        Exibe um resumo dos dados, adaptando-se para DataFrames (2D) ou Arrays NumPy (2D/3D).
        """
        print(f"\n" + "="*40)
        print(f"📊 VISUALIZAÇÃO: {name.upper()}")
        print("="*40)

        # ---------------------------------------------------------
        # 1. SE OS DADOS FOREM PANDAS DATAFRAME (2D Clássico)
        # ---------------------------------------------------------
        if isinstance(data, pd.DataFrame):
            print(f"Formato: Pandas DataFrame | Shape: {data.shape[0]} linhas x {data.shape[1]} colunas\n")
            
            print(f"--- Primeiras {n_rows} linhas ---")
            print(data.head(n_rows))
            
            print(f"\n--- Colunas e Nulos ---")
            info_tab = pd.DataFrame({
                'Tipo': data.dtypes,
                'Nulos (Qtd)': data.isnull().sum(),
                'Nulos (%)': (data.isnull().sum() / len(data) * 100).round(2)
            })
            print(info_tab)

            cols_to_check = [c for c in self.features if c in data.columns]
            if cols_to_check:
                print("\n--- Estatísticas Básicas ---")
                print(data[cols_to_check].describe().T[['mean', 'min', '50%', 'max', 'std']])

        # ---------------------------------------------------------
        # 2. SE OS DADOS FOREM NUMPY ARRAY (Matrizes Pós-Scaler/3D)
        # ---------------------------------------------------------
        elif isinstance(data, np.ndarray):
            print(f"Formato: NumPy Array | Dimensões: {data.ndim}D | Shape: {data.shape}")
            
            # Se for 3D (Ex: X_train para o LSTM)
            if data.ndim == 3:
                amostras, passos, features = data.shape
                print(f"Detalhamento: {amostras} Amostras | {passos} Passos no Tempo | {features} Features\n")
                print(f"--- Amostra do 1º Bloco Temporal (X[0] - Primeiros {n_rows} passos) ---")
                print(data[0][:n_rows])
            
            # Se for 1D ou 2D (Ex: y_train)
            else:
                print(f"\n--- Primeiras {n_rows} linhas ---")
                print(data[:n_rows])

        else:
            print("❌ Formato não suportado. Envie um DataFrame ou NumPy Array.")
            
        print("="*40 + "\n")

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
                df_cleaned['Fase'] = pd.to_numeric(df_cleaned['Fase'].str[0], errors='coerce').fillna(0).astype(int)
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

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforma os dados 2D em sequências 3D para o LSTM.
        Shape de saída: (amostras, sequence_length, features)
        """
        X, y = [], []
        
        # O loop vai até onde é possível criar uma sequência completa
        for i in range(len(features) - self.sequence_length):
            # O 'X' pega um bloco de dados do tamanho da sequência (ex: 3 anos/meses seguidos)
            X.append(features[i:(i + self.sequence_length)])
            
            # O 'y' (target) pega exatamente o próximo valor logo após essa sequência
            y.append(target[i + self.sequence_length])
            
        X_3d = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"Dados transformados para 3D. Shape X: {X_3d.shape} | Shape y: {y_array.shape}")
        
        return X_3d, y_array
    
    def save_preprocessor(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Preprocessador salvo em {path}")

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

    print(df_feature_scaled)
    print(df_target_scaled)
    
    X_train, y_train = preprocessor.create_sequences(features_scaled, target_scaled)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # preprocessor.data_show(cleaned_df, name="DataFrame Limpo e Encoded")
    preprocessor.data_show(X_train, name="X_train (Entrada 3D do LSTM)")
    preprocessor.data_show(y_train, name="y_train (Target 2D)")
    
    print(SCALERS_DIR)
    preprocessor.save_preprocessor(SCALERS_DIR / f"preprocessor_{preprocessor.scaler_type}.joblib")
    

    

if __name__ == "__main__":
    main()
