import os
import requests
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import torch
import apimoex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dqn.nn_module import DQNAgent
from config import Config


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_candles(ticker: str, interval: str = '60', start: int = None):
    with requests.Session() as session:
        return pd.DataFrame(
            apimoex.get_market_candles(
                session=session,
                security=ticker,
                interval=interval,
                start=start,
            )
        )


def get_first_train_dt(train_size: int = 24*365):
    last_dt = datetime.now(ZoneInfo('Europe/Moscow'))
    first_train_dt = last_dt - timedelta(hours=train_size)
    return first_train_dt.strftime("%Y-%m-%d %H:%M:%S")


def get_train_df(config: Config, train_start_dt: str = None):
    print('Loading train data...')
    if train_start_dt is None:
        train_start_dt = get_first_train_dt(train_size=config.train_size)
        
    train_df = get_candles(
        ticker=config.ticker,
        interval=config.interval,
        start=train_start_dt,
    )
    return train_df


def get_last_candle(config: Config):
    start_dt = datetime.now(ZoneInfo('Europe/Moscow'))
    start_dt -= timedelta(hours=2)
    start_dt = start_dt.strftime("%Y-%m-%d %H:%M:%S")

    return get_candles(
        ticker=config.ticker,
        interval=config.interval,
        start=start_dt
    ).iloc[-1]
    

def preprocess_data(dataframe: pd.DataFrame, ema_span: int = 5):
    print('Preprocess train data...')
    dataframe = (dataframe
        .assign(close_ema = lambda _df: _df.close.ewm(span=ema_span, adjust=False).mean())
        .assign(open_ema = lambda _df: _df.open.ewm(span=ema_span, adjust=False).mean())
        .assign(high_ema = lambda _df: _df.high.ewm(span=ema_span, adjust=False).mean())
        .assign(low_ema = lambda _df: _df.low.ewm(span=ema_span, adjust=False).mean())
    )
    dataframe = get_diff_df(dataframe, cols=['close_ema', 'open_ema', 'high_ema', 'low_ema'])
    return dataframe


def get_candles(ticker: str, interval: str = '60', start: int = None):
    with requests.Session() as session:
        return pd.DataFrame(
            apimoex.get_market_candles(
                session=session,
                security=ticker,
                interval=interval,
                start=start,
            )
        )


def fill_memory(agent: DQNAgent, env, config: Config, silent: bool = False):
    if not silent:
        print('Filling model memory...')
    while len(agent.memory) < config.memory_capacity:
        state = env.reset()
        done = False
        while not done:
            action, q_value = agent.act(state, config.device)
            next_state, reward, done = env.step(action, q_value)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
    return agent, env


def train_dqn(agent, env, config, silent: bool = False, use_best_model: bool = False, save_intermediats: bool = False):
    """Trains a DQN agent on the given environment using the provided configuration. 
    The function returns the best model balance and the trained DQN agent.
    
    Arguments:
        agent (DQNAgent): The DQN agent to be trained.
        env (TradingEnvironment): The trading environment to train on.
        config (dict): A dictionary of configuration parameters for training.
        silent (bool): Whether to print training information or not.
        use_best_model (bool): Whether to load the best model at the end of training.
    
    Returns:
        tuple(float, DQNAgent): The best model balance and the trained DQN agent.
    """
    if not silent:
        print('Training model...')
        
    agent.policy_net.train().to(config.device)
    agent.target_net.train().to(config.device)

    best_model_state = None
    best_model_reward = -1e6    
    beta0 = config.beta
    for episode in range(config.num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, q_value = agent.act(state, config.device)
            next_state, reward, done = env.step(action, q_value)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay(config.device)

        config.beta = beta0 * (2 - 0.99**episode)
        agent.update_target_network()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        if episode % config.log_interval == 0:
            config.lr *= 0.99
            agent.change_lr(config.lr)
            result_balance = env.balance + env.position*env.data.at[env.current_step, 'close']

            if total_reward > best_model_reward:
                best_model_state = agent.policy_net.state_dict()
                best_model_res_balance = result_balance

            if not silent:
                print(f'\nEpisode {episode}/{config.num_episodes}')
                print(f'\tTotal Reward: {total_reward}')
                print(f'\tEpsilon: {agent.epsilon:.2f}')
                print(f'\tNum deals: {len(env.trade_history)}')
                print(f'\tResult balance: {result_balance:.3f}')

            if save_intermediats:
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optim_state_dict': agent.optimizer.state_dict(),
                }, f'saved_models/model_{episode}.pt')
    
    if use_best_model:
        agent.policy_net.load_state_dict(best_model_state)
        agent.target_net.load_state_dict(best_model_state)

    return best_model_res_balance, agent


def plot_actions(close_data, trade_history):
    entry_points = np.array([[it['entry_step_i'], it['entry_price']] for it in trade_history])
    close_points = np.array([[it['close_step_i'], it['close_price']] for it in trade_history])
    # green_actions = [close_data[i] if actions[i] == 0 else None for i in range(len(actions))]
    # red_actions = [close_data[i] if actions[i] == 1 else None for i in range(len(actions))]
    
    plt.figure(figsize=(16,4))
    plt.plot(close_data)
    if entry_points.shape[0] > 0:
        plt.scatter(entry_points[:,0], entry_points[:,1], color='g', marker='^')
        plt.scatter(close_points[:,0], close_points[:,1], color='r', marker='v')
    # plt.plot(green_actions, color='g', marker='^', linestyle='dashed')
    # plt.plot(red_actions, color='r', marker='v', linestyle='dashed')
    plt.show()


def evaluate_dqn(agent, env, data, config, silent=False):
    agent.policy_net.eval().to(config.device)
    agent.target_net.eval().to(config.device)

    actions = []
    env.set_data(data)
    state = env.reset()
    done = False

    while not done:
        action, q_value = agent.act(state, config.device, evaluate=True)
        state, reward, done = env.step(action, q_value)
        actions.append(action)

    result_balance = env.balance + env.position*env.data.at[env.current_step, 'close']
    if abs(env.position > 0):
        env.trade_history[-1]['close_price'] = env.data.close.values[-1]
        env.trade_history[-1]['close_step_i'] = env.current_step

    if not silent:
        trade_profits = []
        for deal in env.trade_history:
            try:
                if deal['position_type'] == 'long':
                    trade_profits.append(100 * (deal['close_price'] - deal['entry_price']) / deal['entry_price'])
                else:
                    trade_profits.append(100 * (deal['entry_price'] - deal['close_price']) / deal['entry_price'])
            except KeyError:
                continue
        
        if len(trade_profits) > 0:
            avg_deal_profit = sum(trade_profits) / len(trade_profits)
        else:
            avg_deal_profit = 0

        print(f'Evaluate')
        print(f'\tEpsilon: {agent.epsilon:.2f}')
        print(f'\tNum deals: {len(env.trade_history)}')
        print(f'\tResult balance: {result_balance:.3f}')
        print(f'\tAvg deal profit: {avg_deal_profit:.3f}')


def get_diff_df(data, cols):
    add_data = {}
    for col_name in cols:
        col_value = data[col_name].values
        add_data[col_name + '_diff'] = (col_value[1:] / col_value[:-1] - 1) * 100

    data = data.iloc[1:].reset_index(drop=True)
    for col_name in add_data:
        data[col_name] = add_data[col_name]
        
    return data