import numpy as np
import pandas as pd

getTime = lambda x: f"{x//100:02}:{x%100:02}:{np.random.randint(0, 59):02}.{np.random.randint(0, 999):03}"

def merge_datetime(df, date_col='issue_date', time_col='issue_time'):
    # Cast date and time column to string
    df[date_col] = df[date_col].apply(lambda x: x.split("T")[0])
    df[time_col] = df[time_col].astype('uint16').apply(getTime)

    # Return combined datetime column
    return pd.to_datetime(df['issue_date'] + ' ' + df['issue_time'])