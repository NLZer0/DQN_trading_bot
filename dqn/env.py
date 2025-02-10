import math
import torch


class Environment:
    def __init__(self, data, config):
        self.data = data
        self.window_size = config.window_size
        self.initial_balance = config.initial_balance
        self.commission = config.comission
        # self.is_short = config.is_short
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

    def open_position(self, current_row, q_value, is_short):
        entry_balance = self.balance
        if not is_short:
            # Открываем лонг: покупаем активы
            self.position = (self.balance / current_row['close']) * 0.9
            self.balance -= (current_row['close'] * self.position) * (1 + self.commission)
        else:
            # Открываем шорт: продаем активы (занимаем и продаем)
            self.position = - (self.balance / current_row['close']) * 0.9  # Отрицательное значение для шорта
            self.balance += (current_row['close'] * abs(self.position)) * (1 - self.commission)
        assert abs(self.position) > 0, 'Cannot open position with zero quantity'
        self.entry_price = current_row['close']
        self.trade_history.append({
            'entry_step_i': self.current_step,
            'entry_price': self.entry_price,
            'entry_balance': entry_balance,
            'predict_q': q_value,
            'position_type': 'short' if is_short else 'long'
        })

    def close_position(self, current_row):
        assert self.position != 0, 'position has not opened'
        if self.position > 0:
            # Закрываем лонг: продаем активы
            self.balance += (self.position * current_row['close']) * (1 - self.commission)
        else:
            # Закрываем шорт: покупаем активы для возврата
            cost = abs(self.position) * current_row['close']
            self.balance -= cost * (1 + self.commission)
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

        deal = None
        if action == 0:  # Open new long position and try to close short
            if self.position < 0:
                self.close_position(current_row)
                deal = self.trade_history[-1]

            # open long position after close previous short
            if self.position == 0:
                self.open_position(current_row, q_value, is_short=False)

        elif action == 1: # Open new short position and try to close long  
            if self.position > 0:
                self.close_position(current_row)
                deal = self.trade_history[-1]
            
            # open short position after close previous long
            if self.position == 0:
                self.open_position(current_row, q_value, is_short=True)

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

        if deal is not None:
            # reward takes into account the commission
            close_p = deal['close_balance']
            entry_p = deal['entry_balance']
            deal_profit = 100 * (close_p - entry_p) / entry_p
            # deal_duration = self.trade_history[-1]['close_step_i'] - self.trade_history[-1]['entry_step_i']
            reward += deal_profit

        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        next_state = self._get_state()
        return next_state, reward, self.done
