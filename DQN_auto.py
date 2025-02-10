import os
import pickle
import pandas as pd

import dqn.utils as dqut 
from dqn.nn_module import DQNAgent
from dqn.env import Environment 
from dqn.config import Config


def load_data(n: int = 5_000):
    data = pd.read_csv(config.data_path)
    return data.iloc[-n:].reset_index(drop=True)


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


def fill_memory(agent: DQNAgent, env: Environment):
    while len(agent.memory) < config.memory_capacity:
        state = env.reset()
        done = False
        while not done:
            action, q_value = agent.act(state, config.device)
            next_state, reward, done = env.step(action, q_value)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
    return agent, env


config = Config()
if __name__ == "__main__":
    init_beta = config.beta
    trade_history = []
    train_pointer = config.train_size
    test_pointer = train_pointer + config.test_size
    
    data = preprocess_data(load_data(n=5_000))
    actual_train_data = data.iloc[:train_pointer].reset_index(drop=True)
    actual_test_data = data.iloc[train_pointer:test_pointer].reset_index(drop=True)
    
    print('-'*30, '\n')

    # train_data = data.iloc[:5000]
    # test_data = data.iloc[5000:].reset_index(drop=True)

    config.beta = init_beta
    env = Environment(data=actual_train_data, config=config)
    agent = DQNAgent(config)
    
    agent, env = fill_memory(agent, env)
    # train_result_balance, agent = dqut.train_dqn(agent, env, config, silent=False, use_best_model=False)
    # agent.save_model('saved_models/btc_short_big.pt')

    # agent.load_model('saved_models/btc_short_big.pt')
    # dqut.evaluate_dqn(agent, env, test_data, config, silent=False)
    # trade_history = env.trade_history

    while test_pointer < len(data):
        print(f'\nTest pointer: {test_pointer}')

        config.beta = init_beta
        env = Environment(data=actual_train_data, config=config)
        agent = DQNAgent(config)
        agent, env = fill_memory(agent, env)
       
        train_result_balance, agent = dqut.train_dqn(agent, env, config, silent=True, use_best_model=True)
        dqut.evaluate_dqn(agent, env, actual_test_data, config, silent=False)

        actual_env_trade_history = env.trade_history
        for it in actual_env_trade_history:
            it['entry_step_i'] += train_pointer
            if 'close_step_i' in it:
                it['close_step_i'] += train_pointer
            it['train_result_balance'] = train_result_balance

        trade_history += actual_env_trade_history
        train_pointer += config.step
        test_pointer += config.step

        # Shift train and test data
        actual_train_data = data.iloc[train_pointer-config.train_size:train_pointer].reset_index(drop=True)
        actual_test_data = data.iloc[train_pointer:test_pointer].reset_index(drop=True)

    profit = 0
    n_deals = 0
    for trade in trade_history:
        try:
            profit += 100 * (trade['close_price'] - trade['entry_price']) / trade['entry_price']
            n_deals += 1
        except:
            continue
        
    if n_deals == 0:
        profit = 0
    else:
        profit /= n_deals

    print_results(profit, len(trade_history))
    save_results(profit, len(trade_history), test_name=config.test_name)