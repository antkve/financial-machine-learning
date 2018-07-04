from poloniex import Poloniex
import pandas as pd
from numpy import sign

def get_trade_hist(polo, currencypair, startiso, endiso):
    hist = polo.marketTradeHist(
            'BTC_ETH',
            start="1485954000",
            end="1486018800"
        )[::-1]
    return pd.DataFrame(hist)

# Takes in a series of prices, a threshold for 
# the imbalance counter, and a threshold for the imbalance.
# Returns a series of bars.

def imbalance_bars(df, counter_threshold,
       T_EMA_span, b_EMA_span, target_col='rate'):
    
    bars = []
    imbalance = 0
    T_EMA_mult = 2/(T_EMA_span + 1)
    b_EMA_mult = 2/(b_EMA_span + 1)
    prev_val = df[target_col][0]
    b_EMA = 0.5
    T_EMA = 0
    T = 0

    while True:
        ix, row = next(df.itertuples())
        val = row[target_col]
        diff = val - prev_val
        
        if abs(diff) > counter_threshold * rate:
            b = sign(diff)
        else:
            b = 0
        
        imbalance += b
        P_b_EMA = ((1 + b)/2 - b_EMA) * b_mult + P_b_EMA
        
        T += 1

        if abs(imbalance) >= T_EMA * abs((2 * P_b_EMA - 1)):
            bar_curr = {'end':ix,
                    'span':T,

            
            bars.append(bar_curr)    
            T_EMA = (bars[-1]['span'] - T_EMA) * t_mult + T_EMA
            T = 0

        
        
    

def __main__(currencypair, start, end:
    polo = Poloniex()
    df = pd.DataFrame(
        get_trade_hist(polo, 'BTC_ETH', start, end),
        columns=['amount', 'type', 'globalTradeID', 
            'date', 'rate', 'tradeID', 'total']
        ).set_index('date')
    imbalance_bars = imbalance_bars(df['rate'], 01)
