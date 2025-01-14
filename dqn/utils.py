import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# Training loop
def train_dqn(agent, env, config, silent=False, use_best_model=False):
    agent.policy_net.train().to(config.device)
    agent.target_net.train().to(config.device)

    best_model_state = None
    best_model_reward = -1e6    
    
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

        agent.update_target_network()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        if episode % 10 == 0:
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

            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            torch.save({
                'model_state_dict': agent.policy_net.state_dict(),
                'optim_state_dict': agent.optimizer.state_dict(),
            }, f'saved_models/model_{episode}.pt')
    
    if use_best_model:
        agent.policy_net.load_state_dict(best_model_state)
        agent.target_net.load_state_dict(best_model_state)

    return best_model_res_balance


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
        # if (q_value < 5) & (action == 0):
        #     action = 2
        state, reward, done = env.step(action, q_value)
        actions.append(action)

    result_balance = env.balance + env.position*env.data.at[env.current_step, 'close']
    if env.position > 0:
        env.trade_history[-1]['close_price'] = env.data.close.values[-1]
        env.trade_history[-1]['close_step_i'] = env.current_step

    if not silent:
        try:
            trade_profits = [100*(it['close_price']-it['entry_price']) / it['close_price'] for it in env.trade_history]
            avg_deal_profit = sum(trade_profits) / len(trade_profits)
        except: 
            trade_profits = 0
            avg_deal_profit = 0

        print(f'Evaluate')
        print(f'\tEpsilon: {agent.epsilon:.2f}')
        print(f'\tNum deals: {len(env.trade_history)}')
        print(f'\tResult balance: {result_balance:.3f}')
        print(f'\tAvg deal profit: {avg_deal_profit:.3f}')
        # plot_actions(env.data['close'], env.trade_history)


def get_diff_df(data, cols):
    add_data = {}
    for col_name in cols:
        col_value = data[col_name].values
        add_data[col_name + '_diff'] = (col_value[1:] / col_value[:-1] - 1) * 100

    data = data.iloc[1:].reset_index(drop=True)
    for col_name in add_data:
        data[col_name] = add_data[col_name]
        
    return data