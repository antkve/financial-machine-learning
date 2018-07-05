from poloniex import Poloniex
import pandas as pd
from numpy import sign
from more_itertools import peekable
import datetime as dt
import dateutil.parser as dp
import matplotlib.pyplot as plt
from finance import Bar, Transaction


def get_trade_hist(polo, currencypair, startiso, endiso):
    parsed_start = dp.parse(startiso)
    parsed_end = dp.parse(endiso)
    hist = polo.marketTradeHist(
            currencypair,
            start=parsed_start.strftime('%s'),
            end=parsed_end.strftime('%s'),
        )[::-1]
    ret = [Transaction(row) for row in hist]
    return ret


def time_bars(txs, start, sep_secs):
    txs_iter = peekable(iter(txs))

    end_date = txs[-1].date
    date = txs[0].date
    bars = []
    while date < end_date: 
        date = date + dt.timedelta(seconds=sep_secs)
        if txs_iter.peek().date < date and date < end_date:
            bar = Bar(next(txs_iter), start=date)
            while txs_iter.peek().date < date:
                tx = next(txs_iter)
                bar.update(tx)
            bar.finish(tx)
            bars.append(bar)

    bar.update(tx)
    bar.finish(tx)
    bars.append(bar)
    return bars

def tick_bars(txs, sep):
    txs_iter = iter(txs)

    bars = []

    while True:
        try:
            bar = Bar(next(txs_iter))
            while bar.ticks < sep:
                tx = next(txs_iter)
                bar.update(tx)
            bar.finish(tx)
            bars.append(bar)
        except StopIteration:
            bar.update(tx)
            bar.finish(tx)
            bars.append(bar)
            return bars


# Takes in a series of prices, a threshold for 
# the imbalance counter trigger, and the spans
# for each EMA. Returns a list of bars.

def imbalance_bars(txs, counter_threshold,
       T_EMA_span, b_EMA_span):
    bars = []
    T_EMA_mult = 2/(T_EMA_span + 1)
    E_b_EMA_mult = 2/(b_EMA_span + 1)
    E_b_EMA = 0.5
    T_EMA = 20
    print(txs)
    prev_rate = txs[0].rate
    diff = 0
    txs_iter = iter(txs)

    while True:
        try:
            imbalance = 0
            bar = Bar(next(txs_iter))
            while abs(imbalance) <= T_EMA * abs((2 * E_b_EMA - 1)):
                tx = next(txs_iter)
                bar.update(tx)
                diff = tx.rate - prev_rate
                if abs(diff) > counter_threshold * prev_rate:
                    b = sign(diff)
                else:
                    b = 0
                imbalance += b
                prev_rate = tx.rate
                E_b_EMA = ((1 + b)/2 - E_b_EMA) * E_b_EMA_mult + E_b_EMA
            
            T_EMA = (bar.ticks - T_EMA) * T_EMA_mult + T_EMA
            bar.finish(tx)
            bars.append(bar)

        except StopIteration:
            bar.update(tx)
            bar.finish(tx)
            bars.append(bar)
            return bars

        
def graph(currencypair, start, end):
    polo = Poloniex()
    txs = get_trade_hist(polo, 'BTC_ETH', start, end)
    imbalancebars = imbalance_bars(txs, 0.001, 10, 10)
    timebars = time_bars(txs, start, 200)
    imbalance_y = [bar.open for bar in imbalancebars]
    imbalance_x = [bar.start for bar in imbalancebars]
    time_y = [bar.open for bar in timebars]
    time_x = [bar.start for bar in timebars]

    plt.plot(imbalance_x, imbalance_y, 'b')
    plt.plot(time_x, time_y, 'r')
    plt.show()


if __name__ == '__main__':
    graph('BTC_ETH', '2017-05-03T04:00:00', '2017-05-03T16:00:00')
