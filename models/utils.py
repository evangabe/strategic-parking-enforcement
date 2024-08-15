from math import sqrt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-notebook")
import seaborn as sns


def adfuller_test(ts, p_threshold=0.05):
    result = adfuller(ts)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for val, label in zip(result, labels):
        print(f"{label}: {val}")
    if result[1] <= p_threshold:
        print("Data is stationary => timeseries rejects ADF null hypothesis")
    else:
        print("Data is non-stationary => timeseries does not reject null hypothesis")

def create_weekly_heatmap(ts, title_prefix='[Hollywood]'):
    plt.figure(figsize=(12,12))
    sns.heatmap(
        ts.groupby([ts.index.hour,ts.index.weekday]).mean().unstack(), 
        xticklabels = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'], 
        linewidths=1
    )
    plt.title(f'{title_prefix} Average Citation Count',fontdict={'fontsize':20})
    plt.ylabel('Hour')
    plt.xlabel('Day of Week')

def rmse(true, pred):
    return sqrt(mean_squared_error(true, pred))

def print_rmse_mae(train_y_true, train_y_pred, test_y_true, test_y_pred):
    print(f"Training Set RMSE: {rmse(train_y_true, train_y_pred)}")
    print(f"Testing Set RMSE: {rmse(test_y_true, test_y_pred)}")

def train_test_split(ts, split=0.8):
    split_index = int(ts.shape[0] * split)
    split_date = ts.index[split_index]
    return ts.iloc[:split_index], ts.iloc[split_index:], split_date