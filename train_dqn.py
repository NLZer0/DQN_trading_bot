import argparse
from dqn import utils
from dqn.env import Environment
from dqn.nn_module import DQNAgent
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model_path', help='The path to save the model')
args = parser.parse_args()

config = Config()
if __name__ == '__main__':
    train_df = utils.preprocess_data(utils.get_train_df(config), config.ema_span)
    env = Environment(data=train_df, config=config)
    agent = DQNAgent(config)
    agent, env = utils.fill_memory(agent, env, config)
    
    train_result_balance, agent = utils.train_dqn(
        agent=agent,
        env=env,
        config=config,
        silent=False,
        use_best_model=True,
        save_intermediats=False
    )
    
    folder = '/'.join(args.model_path.split('/')[:-1])
    utils.check_folder(folder)
    agent.save_model(args.model_path)