import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, Optional, List, Dict
import joblib
import logging
import warnings
from src.data.validator import DataValidator, DataCleaner, validate_and_clean
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
        validate_data: bool = True,
        auto_clean: bool = True
    ):
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.validate_data = validate_data
        self.auto_clean = auto_clean

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

    # def validate_and_clean_data(
    #     self,
    #     df: pd.DataFrame
    # ) -> pd.DataFrame:
    #     """
    #     Valida e limpa os dados antes do processamento.

    #     Args:
    #         df: DataFrame com dados brutos
    #         ticker: Simbolo da acao

    #     Returns:
    #         DataFrame validado e limpo
    #     """
    #     original_len = len(df)

    #     if not self.validate_data:
    #         logger.info("Validacao desabilitada, pulando...")
    #         return df

    #     logger.info("=" * 50)
    #     logger.info("VALIDACAO E LIMPEZA DE DADOS")
    #     logger.info("=" * 50)

    #     # Validar e limpar
    #     df_clean, report = validate_and_clean(
    #         df,
    #         auto_clean=self.auto_clean,
    #         verbose=True
    #     )

    #     self.data_quality_report = report

    #     # Estatisticas de limpeza
    #     self.cleaning_stats = {
    #         "original_records": original_len,
    #         "final_records": len(df_clean),
    #         "removed_records": original_len - len(df_clean),
    #         "removal_percentage": ((original_len - len(df_clean)) / original_len) * 100,
    #         "is_valid": report.is_valid,
    #         "warnings": len(report.warnings),
    #         "errors": len(report.errors)
    #     }

    #     if not report.is_valid:
    #         logger.warning("ATENCAO: Dados tem problemas de qualidade!")
    #         logger.warning("O modelo pode ter performance reduzida.")

    #     return df_clean

    # def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Versao SIMPLIFICADA: Apenas retorna os dados limpos, sem criar indicadores.
    #     Isso imita o comportamento do codigo 'vencedor' original.
    #     """
    #     # 1. Validar e limpar (mantemos isso pois e seguranca basica)
    #     df = self.validate_and_clean_data(df)

    #     available_features = [col for col in self.features if col in df.columns]
    #     return df[available_features]

    # def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
    #     """Calcula o RSI (Relative Strength Index)."""
    #     delta = prices.diff()
    #     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    #     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    #     rs = gain / loss
    #     rsi = 100 - (100 / (1 + rs))
    #     return rsi

    # def _calculate_macd(
    #     self,
    #     prices: pd.Series,
    #     fast: int = 12,
    #     slow: int = 26
    # ) -> pd.Series:
    #     """Calcula o MACD."""
    #     ema_fast = prices.ewm(span=fast, adjust=False).mean()
    #     ema_slow = prices.ewm(span=slow, adjust=False).mean()
    #     return ema_fast - ema_slow

    # def _calculate_bollinger_bands(
    #     self,
    #     prices: pd.Series,
    #     period: int = 20,
    #     std_dev: int = 2
    # ) -> Tuple[pd.Series, pd.Series]:
    #     """Calcula as Bandas de Bollinger."""
    #     ma = prices.rolling(window=period, min_periods=1).mean()
    #     std = prices.rolling(window=period, min_periods=1).std()
    #     upper = ma + (std * std_dev)
    #     lower = ma - (std * std_dev)
    #     return upper, lower

    # def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
    #     """
    #     Calcula o ATR (Average True Range).
    #     Mede a volatilidade do mercado.
    #     """
    #     high = df['High']
    #     low = df['Low']
    #     close = df['Close'].shift(1)

    #     tr1 = high - low
    #     tr2 = abs(high - close)
    #     tr3 = abs(low - close)

    #     tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    #     atr = tr.rolling(window=period, min_periods=1).mean()

    #     return atr

    # def _calculate_stochastic(
    #     self,
    #     df: pd.DataFrame,
    #     k_period: int = 14,
    #     d_period: int = 3
    # ) -> Tuple[pd.Series, pd.Series]:
    #     """
    #     Calcula o Stochastic Oscillator.
    #     %K e %D indicam sobrecompra/sobrevenda.
    #     """
    #     low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
    #     high_max = df['High'].rolling(window=k_period, min_periods=1).max()

    #     # Evitar divisao por zero
    #     denominator = high_max - low_min
    #     denominator = denominator.replace(0, np.nan)

    #     stoch_k = ((df['Close'] - low_min) / denominator) * 100
    #     stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()

    #     return stoch_k, stoch_d

    # def _handle_feature_issues(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Trata problemas comuns em features calculadas.

    #     - Valores infinitos
    #     - NaN isolados
    #     - Valores extremos
    #     """
    #     numeric_cols = df.select_dtypes(include=[np.number]).columns

    #     for col in numeric_cols:
    #         # Substituir infinitos por NaN
    #         df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    #         # Interpolar NaN isolados (nao nas bordas)
    #         if df[col].isna().sum() > 0 and df[col].isna().sum() < len(df) * 0.1:
    #             df[col] = df[col].interpolate(method='linear', limit_direction='both')

    #         # Clipar valores extremos (alem de 5 desvios padrao)
    #         if col not in ['Close', 'Open', 'High', 'Low', 'Volume']:
    #             mean = df[col].mean()
    #             std = df[col].std()
    #             if std > 0:
    #                 lower = mean - 5 * std
    #                 upper = mean + 5 * std
    #                 df[col] = df[col].clip(lower, upper)

    #     return df

    # def fit_scalers(self, df: pd.DataFrame) -> None:
    #     """
    #     Ajusta os scalers com os dados de treino.

    #     Args:
    #         df: DataFrame com dados de treino
    #     """
    #     # Selecionar features disponiveis
    #     available_features = [f for f in self.features if f in df.columns]

    #     # Fit do scaler de features
    #     self.feature_scaler.fit(df[available_features].values)

    #     # Fit do scaler do target
    #     self.target_scaler.fit(df[[self.target]].values)

    #     self.is_fitted = True
    #     self.fitted_features = available_features
    #     logger.info(f"Scalers ajustados com {len(available_features)} features")

    # def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Transforma os dados usando os scalers.

    #     Args:
    #         df: DataFrame com dados

    #     Returns:
    #         Tuple com features e target normalizados
    #     """
    #     if not self.is_fitted:
    #         raise ValueError("Scalers nao ajustados. Execute fit_scalers primeiro.")

    #     features_scaled = self.feature_scaler.transform(
    #         df[self.fitted_features].values
    #     )
    #     target_scaled = self.target_scaler.transform(
    #         df[[self.target]].values
    #     )

    #     return features_scaled, target_scaled

    # def create_sequences(
    #     self,
    #     features: np.ndarray,
    #     target: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Cria sequencias para treinamento do LSTM.

    #     Args:
    #         features: Array de features normalizadas
    #         target: Array de target normalizado

    #     Returns:
    #         Tuple com X (sequencias) e y (targets)
    #     """
    #     X, y = [], []

    #     for i in range(self.sequence_length, len(features)):
    #         X.append(features[i - self.sequence_length:i])
    #         y.append(target[i])

    #     return np.array(X), np.array(y)

    # def train_test_split(
    #     self,
    #     X: np.ndarray,
    #     y: np.ndarray,
    #     train_ratio: float = TRAIN_TEST_SPLIT
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Divide os dados em treino e teste (sem shuffle para series temporais).

    #     Args:
    #         X: Features
    #         y: Target
    #         train_ratio: Proporcao de dados para treino

    #     Returns:
    #         X_train, X_test, y_train, y_test
    #     """
    #     split_idx = int(len(X) * train_ratio)

    #     X_train = X[:split_idx]
    #     X_test = X[split_idx:]
    #     y_train = y[:split_idx]
    #     y_test = y[split_idx:]

    #     logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    #     return X_train, X_test, y_train, y_test

    # def inverse_transform_predictions(
    #     self,
    #     predictions: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Reverte a normalizacao das previsoes.

    #     Args:
    #         predictions: Previsoes normalizadas

    #     Returns:
    #         Previsoes na escala original
    #     """
    #     return self.target_scaler.inverse_transform(predictions)

    # def prepare_prediction_data(
    #     self,
    #     df: pd.DataFrame
    # ) -> np.ndarray:
    #     """
    #     Prepara dados para fazer previsoes (ultimos N dias).

    #     Args:
    #         df: DataFrame com dados recentes

    #     Returns:
    #         Array formatado para previsao
    #     """
    #     if not self.is_fitted:
    #         raise ValueError("Scalers nao ajustados. Carregue um modelo treinado.")

    #     # Garantir que temos dados suficientes
    #     if len(df) < self.sequence_length:
    #         raise ValueError(
    #             f"Necessario pelo menos {self.sequence_length} registros, "
    #             f"recebido {len(df)}"
    #         )

    #     # Usar os ultimos sequence_length registros
    #     recent_data = df.tail(self.sequence_length)

    #     # Transformar
    #     features_scaled = self.feature_scaler.transform(
    #         recent_data[self.fitted_features].values
    #     )

    #     # Formatar para LSTM: (1, sequence_length, n_features)
    #     return features_scaled.reshape(1, self.sequence_length, -1)

    # def save_scalers(self, ticker: str) -> None:
    #     """Salva os scalers treinados."""
    #     joblib.dump(
    #         self.feature_scaler,
    #         SCALERS_DIR / f"{ticker}_feature_scaler.pkl"
    #     )
    #     joblib.dump(
    #         self.target_scaler,
    #         SCALERS_DIR / f"{ticker}_target_scaler.pkl"
    #     )
    #     joblib.dump(
    #         self.fitted_features,
    #         SCALERS_DIR / f"{ticker}_features_list.pkl"
    #     )
    #     logger.info(f"Scalers salvos em {SCALERS_DIR}")

    # def load_scalers(self, ticker: str) -> None:
        """Carrega scalers salvos."""
        self.feature_scaler = joblib.load(
            SCALERS_DIR / f"{ticker}_feature_scaler.pkl"
        )
        self.target_scaler = joblib.load(
            SCALERS_DIR / f"{ticker}_target_scaler.pkl"
        )
        self.fitted_features = joblib.load(
            SCALERS_DIR / f"{ticker}_features_list.pkl"
        )
        self.is_fitted = True
        logger.info(f"Scalers carregados de {SCALERS_DIR}")


def main():
    """Exemplo de uso do preprocessador com validacao."""
    caminho_script = Path(__file__).resolve().parent
    raiz_projeto = caminho_script.parents[1]
    caminho_arquivo = raiz_projeto / 'data' / 'datathon_pede_2024.xlsx'
    df = pd.read_excel(caminho_arquivo, sheet_name='PEDE2024')

    print(f"\nDados coletados: {len(df)} registros")

    # Preprocessar COM validacao
    preprocessor = DataPreprocessor(
        features=FEATURES,
        validate_data=True,
        auto_clean=True,
        scaler_type="minmax"
    )
    
    new_df = preprocessor.filter_features(df)
    
    print(preprocessor.data_show(new_df))

    # # Preparar features (inclui validacao automatica)
    # df_features = preprocessor.prepare_features(df, ticker="NVDA")

    # print(f"\nFeatures criadas: {len(df_features.columns)} colunas")
    # print(f"Lista de features:")
    # for i, col in enumerate(df_features.columns, 1):
    #     print(f"   {i:2d}. {col}")

    # # Estatisticas de limpeza
    # if preprocessor.cleaning_stats:
    #     stats = preprocessor.cleaning_stats
    #     print(f"\nEstatisticas de Limpeza:")
    #     print(f"   Registros originais: {stats['original_records']}")
    #     print(f"   Registros finais: {stats['final_records']}")
    #     print(f"   Removidos: {stats['removed_records']} ({stats['removal_percentage']:.2f}%)")

    # # Fit e transform
    # preprocessor.fit_scalers(df_features)
    # features_scaled, target_scaled = preprocessor.transform(df_features)

    # print(f"\nDados normalizados:")
    # print(f"   Features shape: {features_scaled.shape}")
    # print(f"   Target shape: {target_scaled.shape}")
    # print(f"   Features range: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")

    # # Criar sequencias
    # X, y = preprocessor.create_sequences(features_scaled, target_scaled)
    # print(f"\nSequencias criadas:")
    # print(f"   X shape: {X.shape} (samples, timesteps, features)")
    # print(f"   y shape: {y.shape}")

    # # Split
    # X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
    # print(f"\nTrain/Test Split:")
    # print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    # print(f"   Test: X={X_test.shape}, y={y_test.shape}")

    # print("\n" + "=" * 60)
    # print("Preprocessamento concluido com sucesso!")
    # print("=" * 60)


if __name__ == "__main__":
    main()
