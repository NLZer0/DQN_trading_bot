import math
import torch


class Environment:
    def __init__(self, data, config):
        self.data = data
        self.window_size = config.window_size
        self.initial_balance = config.initial_balance
        self.comission = config.comission

        # self.tp = config.tp
        # self.sl = config.sl

    def reset(self):
        self.trade_history = []
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        row = self.data.loc[self.current_step-self.window_size+1:self.current_step]

        current_profit = 0
        if self.position > 0:
            current_profit = 100 * (self.data.at[self.current_step, 'close']-self.entry_price) / self.entry_price

        state = [
            torch.FloatTensor(row
                [[
                    'close_ema_diff',
                    'open_ema_diff',
                    'high_ema_diff',
                    'low_ema_diff',
                    'dn_signal',
                    'up_signal'
                ]].values), 
            torch.FloatTensor([int(self.position > 0), current_profit]),
        ]
        return state
    
    def set_data(self, data):
        self.data = data

    def open_position(self, current_row, q_value):
        entry_balance = self.balance
        self.position = self.balance // current_row['close']
        self.balance -= (current_row['close']*self.position) * (1 + self.comission)
        self.entry_price = current_row['close']
        self.trade_history.append({
            'entry_step_i': self.current_step,
            'entry_price': self.entry_price,
            'entry_balance': entry_balance,
            'predict_q': q_value,
        })

    def close_position(self, current_row):
        self.balance += (self.position * current_row['close']) * (1-self.comission)
        self.position = 0
        self.trade_history[-1]['close_step_i'] = self.current_step
        self.trade_history[-1]['close_price'] = current_row['close']
        self.trade_history[-1]['close_balance'] = self.balance

    def step(self, action, q_value):
        """
        action: 0 - Open position , 1 - Close position, 2 - Do nothing
        """
        reward = 0
        current_row = self.data.loc[self.current_step]

        close_position_flag = False
        if action == 0:  # Try to open new position after closing previous 
            if self.position == 0:
                self.open_position(current_row, q_value)
        if action == 1:  # Close position
            if (self.position > 0) & (not close_position_flag):
                self.close_position(current_row)
                close_position_flag = True

        # check on sl and tp
        # if len(self.trade_history) > 0:
        #     if 'close_balance' not in self.trade_history[-1]: # last position is not closed
        #         current_profit = current_row['close'] / self.trade_history[-1]['entry_price']
        #         current_profit = (current_profit-1) * 100 # normalize to % of profit
        #         if current_profit < self.sl:
        #             self.close_position(current_row)
        #             close_position_flag = True

        #         elif current_profit > self.tp:
        #             self.close_position(current_row)
        #             close_position_flag = True

        if close_position_flag:
            # reward takes into account the commission
            close_p = self.trade_history[-1]['close_balance']
            entry_p = self.trade_history[-1]['entry_balance']
            deal_profit = 100 * (close_p - entry_p) / entry_p
            # deal_duration = self.trade_history[-1]['close_step_i'] - self.trade_history[-1]['entry_step_i']
            reward += deal_profit

        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        next_state = self._get_state()
        return next_state, reward, self.done
