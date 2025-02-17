from dataclasses import dataclass

@dataclass
class Config:
    train_size: int = 24*14 # size in hours
    eval_size: int = int(24*30.5) # size in hours
    interval: int = '60' # api using interval in minutes
    ticker: str = 'SBER'

    ema_span = 5 # коэффициент для вычисления ema
    random_eps = 0.5 # коэффициент случайных действий
    random_eps_scaler = 0.5 # коэффициент затухания случайных действий
    gamma: float = 0.95  # дисконтирование награды
    beta: float = 0.5
    lr: float = 1e-3 # скорость обучения
    # epochs: int = 10  # количество эпох для обновления политики
    device: str = 'cuda'
    tp = 0.02
    sl = 0.02

    input_size: int = 6 # [open, close, high, low, position, current_profit]
    action_size: int = 3  # [купить, продать, удержать]
    hidden_size: int = 128
    num_atoms: int = 51

    silent: bool = False
    num_episodes: int = 30
    log_interval: int = 2
    max_buffer_size: int = 10_000
    batch_size: int = 1024
    comission: float = 0.001
    initial_balance: float = 1_000_000
    window_size: int = 32
    memory_capacity: int = 10_000