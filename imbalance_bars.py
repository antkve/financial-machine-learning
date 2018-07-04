from poloniex import Poloniex
import pandas as pd
from numpy import sign
from more_itertools import peekable
import datetime as dt
import dateutil.parser as dp
import matplotlib.pyplot as plt


def get_trade_hist(polo, currencypair, startiso, endiso):
    parsed_start = dp.parse(startiso)
    parsed_end = dp.parse(endiso)
    hist = polo.marketTradeHist(
            currencypair,
            start=parsed_start.strftime('%s'),
            end=parsed_end.strftime('%s'),
        )[::-1]
    return pd.DataFrame(hist)


class Transaction:
    def __init__(row, dateformat="%Y-%m-%d %H:%M:%s")
        self.rate = float(row.rate)
        self.date = dt.strptime(row.date, dateformat)
        self.amount = float(row.amount)
        self.type = row.type
        self.total = float(row.total)
        

class Bar:
    def __init__(self, tx):
        self.open = self.high = self.low = tx.rate
        self.volume = self.ticks = 0
        self.start = tx.date
        self.txs = [tx]
    
    def update(tx):
        if tx.rate > self.high:
            self.high = tx.rate
        elif tx.rate < self.low:
            self.low = tx.rate
        self.volume += float(tx.amount)
        self.ticks += 1
        self.txs.append(tx)

    def close(tx):
        self.close = tx.rate
        self.end = tx.date


def time_bars(txs, sep_secs):
    txs_iter = iter(txs)

    init_date = txs[0].date
    end_date = txs[-1].date
    date = init_date
    while date < end_date: 
        date = date + dt.timedelta(seconds=sep_secs)
        bar_dates.append(date)

    bars = []
    bar = Bar(tx)

    for start in bardates:

        tx = next(txs_iter)
        bar = Bar(tx)

        
        while 
        if  > bar.start + dt.timedelta(seconds=sep_secs):
            bar.close(tx)
            bars.append(bar)
            try:
                continue
            except StopIteration:
                bars.append(bar)
                return bars
        tx = Transaction(next(df_iter))
        bar.update(tx)


def tick_bars(df, sep):
    df_iter = df.itertuples()

    bars = []
    bar = Bar(Transaction(next(df_iter)))

    while True:
        tx = Transaction(next(df_iter))
        bar.update(tx)
        if bar.ticks >= sep:
            bar.close(tx)
            bars.append(bar)
            try:
                bar = Bar(Transaction(next(df_iter)))
            except StopIteration:
                return bars


# Takes in a series of prices, a threshold for 
# the imbalance counter trigger, and the spans
# for each EMA. Returns a DataFrame of bars.

def imbalance_bars(df, counter_threshold,
       T_EMA_span, b_EMA_span):
    
    bars = []

    imbalance = 0
    T_EMA_mult = 2/(T_EMA_span + 1)
    E_b_EMA_mult = 2/(b_EMA_span + 1)
    prev_rate = float(df['rate'].iloc[0])
    E_b_EMA = 0.5
    T_EMA = T = 20
    start = df['date'].values[0]
    rate = open_curr = float(df['rate'].values[0])
    vol = 0
    diff = 0
    stop = False

    df_iter = peekable(df.itertuples())

    while True:
        try:
            row = next(df_iter)
        except StopIteration: return pd.DataFrame(bars)
        diff = float(row.rate) - rate
        rate = float(row.rate)
        
        if abs(diff) > counter_threshold * rate:
            b = sign(diff)
        else:
            b = 0
        
        imbalance += b
        E_b_EMA = ((1 + b)/2 - E_b_EMA) * E_b_EMA_mult + E_b_EMA

        
        T += 1
        vol += float(row.amount)

        if abs(imbalance) > T_EMA * abs((2 * E_b_EMA - 1)):
            bar = {'start':dt.strptime(start, "%Y-%m-%d %H:%M:%s"),
                    'end':dt.strptime(row.date, "%Y-%m-%d %H:%M:%s")),
                    'span':T,
                    'open':open_curr,
                    'close':rate,
                    'volume':vol}
            bars.append(bar)   

            # Initialize for next bar

            T_EMA = (T - T_EMA) * T_EMA_mult + T_EMA
            T = 0
            vol = 0
            start = df_iter.peek().date
            open_curr = df_iter.peek().rate
            imbalance = 0
            

        
def __main__(currencypair, start, end):
    polo = Poloniex()
    df = pd.DataFrame(
        get_trade_hist(polo, 'BTC_ETH', start, end),
        columns=['amount', 'type', 'globalTradeID', 
            'date', 'rate', 'tradeID', 'total']
        )
    print(df)
    bars = imbalance_bars(df, 0.001, 10, 10, 20)
    plt.plot(df['date'], [float(rate) for rate in df['rate']], 'b')
    plt.plot(bars['start'], [float(open) for open in bars['open']], 'r')
    plt.show()

__main__('BTC_ETH', '2017-05-03T04:00:00', '2017-05-03T04:30:00')
