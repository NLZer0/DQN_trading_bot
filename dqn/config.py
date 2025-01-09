from dataclasses import dataclass

@dataclass
class Config:
    # train_size: int = 24*100
    # test_size: int = int(24*30.5)
    # step: int = int(24*30.5)
    
    train_size: int = 1000
    test_size: int = 100
    step: int = 100
    
    input_size: int = 8 # [open, close, high, low, position, current_profit]
    action_size: int = 3  # [купить, продать, удержать]
    hidden_size: int = 32

    ema_span = 5 # коэффициент для вычисления ema
    random_eps = 0.5 # коэффициент случайных действий
    random_eps_scaler = 0.8 # коэффициент затухания случайных действий
    gamma: float = 0.99  # дисконтирование награды
    lr: float = 1e-3 # скорость обучения
    # epochs: int = 10  # количество эпох для обновления политики
    device: str = 'cuda'     
    
    silent: bool = False
    num_episodes: int = 30
    log_interval: int = 10
    data_path: str = '/home/nikolayz/Рабочий стол/RL_research/data/sber_1h_labeled.csv'
    model_path: str = 'saved_models/qnet/model'
    test_name: str = 'base_dqn'

    max_buffer_size: int = 30_000
    batch_size: int = 1024
    comission: float = 0.0005
    initial_balance: float = 10_000