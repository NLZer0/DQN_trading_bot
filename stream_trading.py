import argparse
from datetime import datetime

from dqn import utils
from dqn.env import Environment
from dqn.nn_module import DQNAgent
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model_path', type=str, help='The path for loading the model')
parser.add_argument('-l', '--log_file_path', type=str, help='Path of log file')
args = parser.parse_args()
config = Config()

if __name__ == '__main__':
    agent = DQNAgent(config)
    env = Environment(data=None, config=config, is_stream=True)
    agent.load_model(args.model_path)
    state = env.reset()

    log_file = open(args.log_file_path, 'w')
    log_file.write(','.join(
        [
            'datetime', 'open', 'close',
            'low', 'high', 'action'
        ]
    ) + '\n')
    
    last_hour = datetime.now().hour
    while True:
        if datetime.now().hour != last_hour:
            print(datetime.now())
            last_hour = datetime.now().hour

            state = env._get_state()
            action, q_value = agent.act(state, config.device, evaluate=True)
            _, reward, _ = env.step(action, q_value)

            last_row = env.last_row[['begin', 'open', 'close', 'low', 'high']].values.tolist()
            last_row.append(action)
            log_file.write(','.join(map(str, last_row)) + '\n')
