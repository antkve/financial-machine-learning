from poloniex import Poloniex
import pandas as pd
from numpy import sign
from more_itertools import peekable
import datetime as dt
import dateutil.parser as dp
import matplotlib.pyplot as plt
import finance
from matplotlib.finance import candlestick_ohlc


def get_trade_hist(polo, currencypair, startiso, endiso):
    parsed_start = dp.parse(startiso)
    parsed_end = dp.parse(endiso)
    hist = polo.marketTradeHist(
            currencypair,
            start=parsed_start.strftime('%s'),
            end=parsed_end.strftime('%s'),
        )[::-1]
    return pd.DataFrame(hist)


def time_bars(txs, start, sep_secs):
    txs_iter = peekable(iter(txs))

    end_date = txs[-1].date
    date = txs[0].date
    bars = []
    while date < end_date: 
        date = date + dt.timedelta(seconds=sep_secs)
        if txs_iter.peek().date < date:
            bar = Bar(next(txs_iter), start=date)
            while txs_iter.peek().date < date:
                tx = next(txs_iter)
                bar.update(tx)
            bar.close(tx)
            bars.append(bar)
    return bars


def tick_bars(txs, sep):
    txs_iter = peekable(iter(txs))

    bars = []

    while True:
        bar = Bar(next(txs_iter))
        while bar.ticks < sep:
            tx = next(txs_iter)
            bar.update(tx)
        bar.close(tx)
        bars.append(bar)
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