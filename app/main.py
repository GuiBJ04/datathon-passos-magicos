# """
# API FastAPI para previsao de precos de acoes com LSTM
# """
# from fastapi import FastAPI, HTTPException, Query, Path
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from datetime import datetime, timedelta
# import logging
# import os
# import numpy as np

# from src.config import DEFAULT_TICKER, API_HOST, API_PORT, MODELS_DIR, SEQUENCE_LENGTH, FEATURES
# from src.data.collector import StockDataCollector
# from src.data.preprocessor import DataPreprocessor
# from src.models.lstm import StockLSTMModel

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class StockPredictor:
#     """Classe para fazer previsoes de acoes."""

#     def __init__(self, ticker: str = DEFAULT_TICKER):
#         self.ticker = ticker.upper()
#         self.model = StockLSTMModel()
#         self.preprocessor = DataPreprocessor(
#             features=FEATURES,
#             validate_data=False,  # Desabilitado por padrao para previsoes
#             auto_clean=False
#         )
#         self.collector = StockDataCollector(self.ticker)
#         self._is_loaded = False

#     def load(self) -> None:
#         """Carrega modelo e scalers treinados."""
#         try:
#             self.model.load_model(self.ticker)
#             self.preprocessor.load_scalers(self.ticker)
#             self._is_loaded = True
#             logger.info(f"Modelo e scalers carregados para {self.ticker}")
#         except FileNotFoundError as e:
#             raise FileNotFoundError(
#                 f"Modelo nao encontrado para {self.ticker}. "
#                 "Execute train.py primeiro."
#             ) from e

#     def predict_next_day(self) -> dict:
#         """Preve o preco de fechamento do proximo dia util."""
#         if not self._is_loaded:
#             self.load()

#         logger.info("Coletando dados recentes...")
#         df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)

#         self.preprocessor.validate_data = False
#         df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)

#         if len(df_features) < SEQUENCE_LENGTH:
#             raise ValueError(
#                 f"Dados insuficientes. Necessario: {SEQUENCE_LENGTH}, "
#                 f"Disponivel: {len(df_features)}"
#             )

#         X = self.preprocessor.prepare_prediction_data(df_features)
#         prediction_scaled = self.model.predict(X)
#         prediction = self.preprocessor.inverse_transform_predictions(prediction_scaled)

#         last_close = float(df_features['Close'].iloc[-1])
#         predicted_price = float(prediction[0][0])
#         change = predicted_price - last_close
#         change_pct = (change / last_close) * 100

#         last_date = df_features.index[-1]
#         next_date = self._get_next_business_day(last_date)

#         return {
#             "ticker": self.ticker,
#             "prediction_date": next_date.strftime("%Y-%m-%d"),
#             "last_close": round(last_close, 2),
#             "last_close_date": last_date.strftime("%Y-%m-%d"),
#             "predicted_close": round(predicted_price, 2),
#             "expected_change": round(change, 2),
#             "expected_change_pct": round(change_pct, 2),
#             "generated_at": datetime.now().isoformat()
#         }

#     def predict_n_days(self, n_days: int = 5) -> list:
#         """Preve os proximos N dias (previsao iterativa)."""
#         if not self._is_loaded:
#             self.load()

#         df = self.collector.get_latest_data(days=SEQUENCE_LENGTH + 30)

#         self.preprocessor.validate_data = False
#         df_features = self.preprocessor.prepare_features(df, ticker=self.ticker)

#         predictions = []
#         current_features = df_features.copy()
#         last_date = current_features.index[-1]

#         for i in range(n_days):
#             X = self.preprocessor.prepare_prediction_data(current_features)
#             pred_scaled = self.model.predict(X)
#             pred = self.preprocessor.inverse_transform_predictions(pred_scaled)[0][0]

#             next_date = self._get_next_business_day(last_date)

#             predictions.append({
#                 "day": i + 1,
#                 "date": next_date.strftime("%Y-%m-%d"),
#                 "predicted_close": round(float(pred), 2)
#             })

#             # Atualizar para proxima previsao (heuristica simples)
#             new_row = current_features.iloc[-1].copy()
#             new_row['Close'] = pred
#             new_row['Open'] = current_features['Close'].iloc[-1]
#             new_row['High'] = max(pred, new_row['Open']) * 1.01
#             new_row['Low'] = min(pred, new_row['Open']) * 0.99
#             new_row.name = next_date

#             current_features = self.preprocessor.prepare_features(
#                 current_features._append(new_row),
#                 ticker=self.ticker
#             )
#             last_date = next_date

#         return predictions

#     def _get_next_business_day(self, date) -> datetime:
#         """Retorna o proximo dia util."""
#         next_day = date + timedelta(days=1)
#         while next_day.weekday() >= 5:
#             next_day += timedelta(days=1)
#         return next_day

#     def get_model_info(self) -> dict:
#         """Retorna informacoes sobre o modelo carregado."""
#         if not self._is_loaded:
#             self.load()

#         stock_info = self.collector.get_stock_info()

#         return {
#             "ticker": self.ticker,
#             "stock_info": stock_info,
#             "model_loaded": True,
#             "sequence_length": getattr(self.preprocessor, "sequence_length", SEQUENCE_LENGTH),
#             "features_used": getattr(self.preprocessor, "fitted_features", FEATURES)
#         }


# # Inicializar FastAPI
# app = FastAPI(
#     title="Stock Price Predictor API",
#     description="""
#     API para previsao de precos de acoes usando redes neurais LSTM.

#     ## Funcionalidades

#     * **Previsao de proximo dia**: Preve o preco de fechamento do proximo dia util
#     * **Previsao de multiplos dias**: Preve os proximos N dias (com incerteza crescente)

#     ## Disclaimer

#     Esta API e apenas para fins educacionais e nao constitui recomendacao de investimento.
#     """,
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Cache de preditores
# predictors = {}


# # Modelos Pydantic
# class PredictionResponse(BaseModel):
#     """Resposta de previsao de um dia."""
#     ticker: str = Field(..., description="Simbolo da acao")
#     prediction_date: str = Field(..., description="Data da previsao")
#     last_close: float = Field(..., description="Ultimo preco de fechamento")
#     last_close_date: str = Field(..., description="Data do ultimo fechamento")
#     predicted_close: float = Field(..., description="Preco previsto")
#     expected_change: float = Field(..., description="Variacao esperada em $")
#     expected_change_pct: float = Field(..., description="Variacao esperada em %")
#     generated_at: str = Field(..., description="Timestamp da previsao")

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "ticker": "NVDA",
#                 "prediction_date": "2024-01-15",
#                 "last_close": 547.89,
#                 "last_close_date": "2024-01-12",
#                 "predicted_close": 552.34,
#                 "expected_change": 4.45,
#                 "expected_change_pct": 0.81,
#                 "direction": "UP",
#                 "generated_at": "2024-01-14T10:30:00"
#             }
#         }


# class MultiDayPrediction(BaseModel):
#     """Previsao para um dia especifico."""
#     day: int = Field(..., description="Numero do dia (1 = amanha)")
#     date: str = Field(..., description="Data da previsao")
#     predicted_close: float = Field(..., description="Preco previsto")


# class MultiDayPredictionResponse(BaseModel):
#     """Resposta de previsao de multiplos dias."""
#     ticker: str
#     predictions: List[MultiDayPrediction]
#     warning: str = Field(
#         default="Previsoes mais distantes tem maior incerteza",
#         description="Aviso sobre incerteza"
#     )
#     generated_at: str


# class ModelInfoResponse(BaseModel):
#     """Informacoes do modelo."""
#     ticker: str
#     stock_info: dict
#     model_loaded: bool
#     sequence_length: int
#     features_used: List[str]


# class HealthResponse(BaseModel):
#     """Resposta de health check."""
#     status: str
#     timestamp: str
#     available_models: List[str]


# class ErrorResponse(BaseModel):
#     """Resposta de erro."""
#     error: str
#     detail: str


# def get_predictor(ticker: str) -> StockPredictor:
#     """Obtem ou cria um predictor para o ticker."""
#     ticker = ticker.upper()

#     if ticker not in predictors:
#         try:
#             predictor = StockPredictor(ticker)
#             predictor.load()
#             predictors[ticker] = predictor
#             logger.info(f"Predictor carregado para {ticker}")
#         except FileNotFoundError:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Modelo nao encontrado para {ticker}. Execute o treinamento primeiro."
#             )
#         except Exception as e:
#             logger.error(f"Erro ao carregar predictor: {e}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Erro ao carregar modelo: {str(e)}"
#             )

#     return predictors[ticker]


# def get_available_models() -> List[str]:
#     """Lista modelos disponiveis."""
#     models = []
#     if MODELS_DIR.exists():
#         for file in MODELS_DIR.glob("*_lstm_model.keras"):
#             ticker = file.stem.replace("_lstm_model", "")
#             models.append(ticker)
#     return models


# # Endpoints
# @app.get(
#     "/",
#     summary="Pagina inicial",
#     response_class=JSONResponse
# )
# async def root():
#     """Pagina inicial da API."""
#     return {
#         "name": "Stock Price Predictor API",
#         "version": "1.0.0",
#         "ticker": DEFAULT_TICKER,
#         "docs": "/docs",
#         "endpoints": {
#             "predict": "/predict",
#             "predict_days": "/predict/days/{n_days}",
#             "health": "/health"
#         }
#     }


# @app.get(
#     "/health",
#     response_model=HealthResponse,
#     summary="Health check"
# )
# async def health_check():
#     """Verifica status da API e modelos disponiveis."""
#     return HealthResponse(
#         status="healthy",
#         timestamp=datetime.now().isoformat(),
#         available_models=get_available_models()
#     )


# @app.get(
#     "/predict",
#     response_model=PredictionResponse,
#     responses={
#         404: {"model": ErrorResponse, "description": "Modelo nao encontrado"},
#         500: {"model": ErrorResponse, "description": "Erro interno"}
#     },
#     summary="Prever proximo dia"
# )
# async def predict_next_day():
#     """
#     Preve o preco de fechamento do proximo dia util para NVDA.

#     Retorna previsao com variacao esperada.
#     """
#     try:
#         predictor = get_predictor(DEFAULT_TICKER)
#         result = predictor.predict_next_day()
#         return PredictionResponse(**result)
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Erro na previsao: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Erro ao fazer previsao: {str(e)}"
#         )


# @app.get(
#     "/predict/days/{n_days}",
#     response_model=MultiDayPredictionResponse,
#     responses={
#         400: {"model": ErrorResponse, "description": "Parametros invalidos"},
#         404: {"model": ErrorResponse, "description": "Modelo nao encontrado"},
#         500: {"model": ErrorResponse, "description": "Erro interno"}
#     },
#     summary="Prever multiplos dias"
# )
# async def predict_multiple_days(
#     n_days: int = Path(
#         ...,
#         ge=1,
#         le=30,
#         description="Numero de dias para prever (1-30)"
#     )
# ):
#     """
#     Preve os precos de fechamento dos proximos N dias uteis para NVDA.

#     - **n_days**: Numero de dias (1-30)

#     **Atencao**: Previsoes mais distantes tem incerteza significativamente maior.
#     """
#     try:
#         predictor = get_predictor(DEFAULT_TICKER)
#         predictions = predictor.predict_n_days(n_days)

#         return MultiDayPredictionResponse(
#             ticker=DEFAULT_TICKER,
#             predictions=[MultiDayPrediction(**p) for p in predictions],
#             generated_at=datetime.now().isoformat()
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Erro na previsao multi-dia: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Erro ao fazer previsao: {str(e)}"
#         )


# # Eventos de startup/shutdown
# @app.on_event("startup")
# async def startup_event():
#     """Evento de inicializacao."""
#     logger.info("Stock Predictor API iniciando...")
#     models = get_available_models()
#     logger.info(f"Modelos disponiveis: {models}")


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Evento de encerramento."""
#     logger.info("Stock Predictor API encerrando...")
#     predictors.clear()


# def main():
#     """Inicia o servidor."""
#     import uvicorn

#     print("\n" + "=" * 60)
#     print("STOCK PREDICTOR API")
#     print("=" * 60)
#     print(f"\nServidor: http://{API_HOST}:{API_PORT}")
#     print(f"Documentacao: http://localhost:{API_PORT}/docs")
#     print(f"ReDoc: http://localhost:{API_PORT}/redoc")
#     print("\n" + "=" * 60)

#     uvicorn.run(
#         "src.api.app:app",
#         host=API_HOST,
#         port=API_PORT,
#         reload=True,
#         log_level="info"
#     )


# if __name__ == "__main__":
#     main()
