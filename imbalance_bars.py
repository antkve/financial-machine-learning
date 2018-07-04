from poloniex import Poloniex
import pandas as pd
from numpy import sign
from more_itertools import peekable
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


# Takes in a series of prices, a threshold for 
# the imbalance counter, and a threshold for the imbalance.
# Returns a series of bars.

def imbalance_bars(df, counter_threshold,
       T_EMA_span, b_EMA_span):
    
    bars = []
    imbalance = 0
    T_EMA_mult = 2/(T_EMA_span + 1)
    E_b_EMA_mult = 2/(b_EMA_span + 1)
    prev_rate = float(df['rate'].iloc[0])
    E_b_EMA = 0.5
    T_EMA = 20
    T = 0

    start_curr = df['date'].values[0]
    rate = open_curr = float(df['rate'].values[0])
    vol_curr = 0
    diff = 0

    df_iter = peekable(df.itertuples())

    while True:
        row = next(df_iter)
        diff = float(row.rate) - rate
        rate = float(row.rate)
        
        print(diff)
        if abs(diff) > counter_threshold * rate:
            b = sign(diff)
        else:
            b = 0
        
        imbalance += b
        # Exponential MA of expected value of b
        E_b_EMA = ((1 + b)/2 - E_b_EMA) * E_b_EMA_mult + E_b_EMA

        
        T += 1
        vol_curr += float(row.amount)

        if abs(imbalance) > T_EMA * abs((2 * E_b_EMA - 1)):
            bar = {'start':start_curr,
                    'end':row.date,
                    'span':T,
                    'open':open_curr,
                    'close':rate,
                    'volume':vol_curr}
            bars.append(bar)   
            print("E_b: {}".format(E_b_EMA))
            print("E_T: {}".format(T_EMA))

            # Initialize for next bar
            vol_curr = 0
            T = 0

            T_EMA = (T - T_EMA) * T_EMA_mult + T_EMA
            try:
                start_curr = df_iter.peek().date
            except StopIteration:
                return pd.DataFrame(bars)
            open_curr = df_iter.peek().rate
            imbalance = 0
            print("rate: {}, prev_close: {}".format(rate, bars[-1]['close']))
            

        
def __main__(currencypair, start, end):
    polo = Poloniex()
    df = pd.DataFrame(
        get_trade_hist(polo, 'BTC_ETH', start, end),
        columns=['amount', 'type', 'globalTradeID', 
            'date', 'rate', 'tradeID', 'total']
        )
    bars = imbalance_bars(df, 0, 10, 10)
    print(df)
    print(bars)
    plt.plot(df['date'], [float(rate) for rate in df['rate']], 'r--')
    plt.plot(bars['start'], [float(open) for open in bars['open']], 'bs')
    plt.show()


__main__('BTC_ETH', '2017-05-02T04:04:00', '2017-05-02T04:10:00')
