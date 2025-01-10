import os
import pickle
import pandas as pd

import dqn.utils as dqut 
from dqn.nn_module import DQNAgent
from dqn.env import Environment 
from dqn.config import Config


def load_data(n: int = 5_000):
    return pd.read_csv(config.data_path).iloc[:n]


def preprocess_data(full_data):
    full_data = (full_data
        .assign(close_ema = lambda _df: _df.close.ewm(span=config.ema_span, adjust=False).mean())
        .assign(open_ema = lambda _df: _df.open.ewm(span=config.ema_span, adjust=False).mean())
        .assign(high_ema = lambda _df: _df.high.ewm(span=config.ema_span, adjust=False).mean())
        .assign(low_ema = lambda _df: _df.low.ewm(span=config.ema_span, adjust=False).mean())         
    )
    full_data = dqut.get_diff_df(full_data, cols=['close_ema', 'open_ema', 'high_ema', 'low_ema'])
    return full_data


def save_results(profit, n_deals, test_name):
    if not os.path.exists(f'results/{test_name}'):
        os.makedirs(f'results/{test_name}')
    
    with open(f'results/{test_name}/metrics.txt', 'w') as f:
        f.write(f'Avg profit per deal: {profit:.3f}%\n')
        f.write(f'Num deals: {n_deals}')

    with open(f'results/{test_name}/deals_history.pkl', 'wb') as f:
        pickle.dump(trade_history, f)


def print_results(profit, n_deals):
    print('\n\n')
    print('-'*30)
    print(f'Avg profit per deal: {profit:.3f}%')
    print(f'Num deals: {n_deals}')


config = Config()
if __name__ == "__main__":
    trade_history = []
    train_pointer = config.train_size
    test_pointer = train_pointer + config.test_size
    
    data = preprocess_data(load_data(n=5000))
    actual_train_data = data.iloc[:train_pointer].reset_index(drop=True)
    actual_test_data = data.iloc[train_pointer:test_pointer].reset_index(drop=True)
    
    print('-'*30, '\n')
    while test_pointer < len(data):
        print(f'\nTest pointer: {test_pointer}')
        env = Environment(data=actual_train_data, initial_balance=10_000, comission=config.comission)
        agent = DQNAgent(config)
        dqut.train_dqn(agent, env, config, silent=True, use_best_model=True)
        dqut.evaluate_dqn(agent, env, actual_test_data, config, silent=False)

        actual_env_trade_history = env.trade_history
        for it in actual_env_trade_history:
            it['entry_step_i'] += train_pointer
            if 'close_step_i' in it:
                it['close_step_i'] += train_pointer

        trade_history += actual_env_trade_history
        train_pointer += config.step
        test_pointer += config.step

        actual_train_data = data.iloc[train_pointer-config.train_size:train_pointer].reset_index(drop=True)
        actual_test_data = data.iloc[train_pointer:test_pointer].reset_index(drop=True)

    balance = 100_000
    profit = 0
    n_deals = 0
    for trade in trade_history:
        try:
            profit += 100 * (trade['close_price'] - trade['entry_price']) / trade['entry_price']
            n_deals += 1
        except:
            continue
    profit /= n_deals

    print_results(profit, len(trade_history))
    save_results(profit, len(trade_history), test_name=config.test_name)