import datetime as dt


class Transaction:
    def __init__(self, row, dateformat="%Y-%m-%d %H:%M:%S"):
        self.rate = float(row['rate'])
        self.date = dt.datetime.strptime(row['date'], dateformat)
        self.amount = float(row['amount'])
        self.type = row['type']
        self.total = float(row['total'])


class Bar:
    def __init__(self, tx, start=None):
        self.open = self.high = self.low = tx.rate
        self.volume = self.ticks = 0
        self.ticks = 0
        if start:
            self.start = start
        else:
            self.start = tx.date
        self.txs = []
        self.update(tx)

    def update(self, tx):
        if tx.rate > self.high:
            self.high = tx.rate
        elif tx.rate < self.low:
            self.low = tx.rate
        self.volume += tx.amount
        self.ticks += 1
        self.txs.append(tx)

    def finish(self, tx):
        self.close = tx.rate
        self.end = tx.date


