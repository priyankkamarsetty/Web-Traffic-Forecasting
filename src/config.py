from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    DATA_PATH: str = "data/web_traffic.csv"
    OUTPUT_DIR: str = "outputs"
    TRAIN_RATIO: float = 0.65
    TIME_STEPS: int = 100
    FORECAST_DAYS: int = 30
    RANDOM_SEED: int = 42

CFG = Config()
