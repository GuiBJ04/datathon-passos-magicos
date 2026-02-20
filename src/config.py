from pathlib import Path

# Diretorios - BASE_DIR aponta para a raiz do projeto (um nivel acima de src/)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SCALERS_DIR.mkdir(exist_ok=True)

# --- PARAMETROS DO GRID SEARCH VENCEDOR ---
SEQUENCE_LENGTH = 30           # look_back=30
LSTM_UNITS_1 = 100             # lstm_units=100
LSTM_UNITS_2 = 100             # (Camada 2 igual a 1)
DENSE_UNITS = 32               # (Nao usado explicitamente no original, mas mantemos)
DROPOUT_RATE = 0.2             # dropout=0.2

# Treinamento
EPOCHS = 100                   # epochs=100
BATCH_SIZE = 32                # batch=32
TRAIN_TEST_SPLIT = 0.8         # 80% treino

FEATURES = ["Defasagem", "Fase", "INDE 2024", "Pedra 2024", "Idade", "Gênero", "Ano ingresso", "Instituição de ensino"]
TARGET = "Defasagem"

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

API_HOST = "0.0.0.0"
API_PORT = 8000
