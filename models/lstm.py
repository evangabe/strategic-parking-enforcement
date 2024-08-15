import pandas as pd

def convert_ts_to_supervised(ts, n_in=1, n_out=1):
    df = pd.DataFrame(ts)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    return pd.concat(cols, axis=1).dropna().values