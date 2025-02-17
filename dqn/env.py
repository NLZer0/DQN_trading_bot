import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import torch
from dqn.utils import get_candles, preprocess_data
from config import Config


class Environment:
    def __init__(self, data, config: Config, is_stream: bool = False):
        self.is_stream = is_stream
        self.data = data
        self.window_size = config.window_size
        self.initial_balance = config.initial_balance
        self.commission = config.comission
        self.tp = config.tp
        self.sl = config.sl
        self.ticker = config.ticker
        self.interval = config.interval
        self.ema_span = config.ema_span
        self.last_row = None
        self.done = False
        self.position = 0
        self.entry_price = 0
        self.trade_history = []
        self.balance = self.initial_balance
        self.current_step = self.window_size

    def reset(self):
        self.trade_history = []
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_last_candle(self):
        start_dt = datetime.now(ZoneInfo('Europe/Moscow'))
        start_dt -= timedelta(hours=self.window_size*5)
        start_dt = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        last_df = preprocess_data(
            get_candles(
                ticker=self.ticker,
                interval=self.interval,
                start=start_dt
            ),
            ema_span=self.ema_span
        )
        return last_df.iloc[-self.window_size:]
        
    def _get_state(self):
        if not self.is_stream:
            row = self.data.loc[self.current_step-self.window_size+1:self.current_step]
        else:
            row = self._get_last_candle()
            
        self.last_row = row.iloc[-1]
        current_price = self.last_row['close']
        
        current_profit = 0
        if self.position > 0:
            current_profit = 100 * (current_price - self.entry_price) / self.entry_price
        elif self.position < 0:
            current_profit = 100 * (self.entry_price - current_price) / self.entry_price

        state = [
            torch.FloatTensor(row
                [[
                    'close_ema_diff',
                    'open_ema_diff',
                    'high_ema_diff',
                    'low_ema_diff',
                ]].values.astype(float)), 
            torch.FloatTensor([float(self.position > 0), current_profit]),
        ]
        return state
    
    def _check_risk(self, current_row):
        if self.position > 0:
            max_price_diff = current_row['high'] - self.entry_price
            min_price_diff = current_row['low'] - self.entry_price
        else:
            max_price_diff = self.entry_price - current_row['low']
            min_price_diff = self.entry_price - current_row['high']
            
        if min_price_diff <= -self.sl * self.entry_price:
            self.close_position(current_row)
            return True
        elif max_price_diff >= self.tp * self.entry_price:
            self.close_position(current_row)
            return True
        
        return False
            
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
        if self.is_stream:
            current_row = self._get_last_candle().iloc[-1]
        else:
            current_row = self.data.loc[self.current_step]

        deal = None
        if action == 0:  # Open new long position and try to close short
            if self.position < 0:
                self.close_position(current_row)
                deal = self.trade_history[-1]

            # open long position after close previous short
            # if self.position == 0:
                # self.open_position(current_row, q_value, is_short=False)

        elif action == 1: # Open new short position and try to close long  
            # if self.position > 0:
                # self.close_position(current_row)
                # deal = self.trade_history[-1]

            # open short position after close previous long
            if self.position == 0:
                self.open_position(current_row, q_value, is_short=True)

        if self.position != 0:
            deal_flag = self._check_risk(current_row)
            if deal_flag:
                deal = self.trade_history[-1]

        if deal is not None:
            # reward takes into account the commission
            close_p = deal['close_balance']
            entry_p = deal['entry_balance']
            deal_profit = 100 * (close_p - entry_p) / entry_p
            # deal_duration = self.trade_history[-1]['close_step_i'] - self.trade_history[-1]['entry_step_i']
            reward += deal_profit

        if self.is_stream:
            return None, reward, None

        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        if self.done and self.position != 0:
            self.close_position(current_row)
            
        next_state = self._get_state()
        return next_state, reward, self.done
