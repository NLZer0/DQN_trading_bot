import os
import pickle
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dqn import utils
from dqn.env import Environment
from dqn.nn_module import DQNAgent
from config import Config


def eval_iteration(actual_train_data, actual_test_data, config: Config):
    env = Environment(data=actual_train_data, config=config)
    agent = DQNAgent(config)
    agent, env = utils.fill_memory(agent, env, silent=True, config=config)
    
    train_result_balance, agent = utils.train_dqn(agent, env, config, silent=True, use_best_model=True)
    utils.evaluate_dqn(agent, env, actual_test_data, config, silent=False)
    print(f'\t{train_result_balance}')
    return env, agent, train_result_balance


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
    

def get_dataframe(test_size: int, config: Config, ):
    last_dt = datetime.now(ZoneInfo('Europe/Moscow'))
    train_start_dt = last_dt - timedelta(hours=(config.train_size+test_size))
    train_start_dt = train_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    dataframe = utils.preprocess_data(utils.get_train_df(config, train_start_dt), config.ema_span)
    return dataframe

    
parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model_path', type=str, help='The path to save the model')
parser.add_argument('-t', '--test_size', type=int, help='Test size in hours')
args = parser.parse_args()

# class Args:
#     def __init__(self):
#         self.test_size = 20_000
#         self.model_path = 'saved_models/long_model.pt'
# args = Args()

config = Config()
if __name__ == '__main__':
    dataframe = get_dataframe(args.test_size, config)
    print(f'Dataframe size: {dataframe.shape[0]}')
    
    train_pointer = int(config.train_size)
    test_pointer = int(train_pointer + config.eval_size)
    actual_train_data = dataframe.iloc[:train_pointer].reset_index(drop=True)
    actual_test_data = dataframe.iloc[train_pointer:test_pointer].reset_index(drop=True)
    trade_history = []
    env = Environment(data=actual_train_data, config=config)
    agent = DQNAgent(config)
    agent, env = utils.fill_memory(agent, env, config, silent=True)
    
    print('-'*30)
    while test_pointer < dataframe.shape[0]:
        print(f'\nTest pointer: {test_pointer}')

        env, agent, train_result_balance = eval_iteration(
            actual_train_data=actual_train_data,
            actual_test_data=actual_test_data, 
            config=config,
        )

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
        actual_train_data = dataframe.iloc[train_pointer-config.train_size:train_pointer].reset_index(drop=True)
        actual_test_data = dataframe.iloc[train_pointer:test_pointer].reset_index(drop=True)

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
    save_results(profit, len(trade_history), test_name='results/moex')