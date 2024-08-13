import os
from dotenv import load_dotenv

load_dotenv()
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
# import plotly.express as px
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-notebook")
import seaborn as sns

# px.set_mapbox_access_token(os.getenv('MAPBOX_TOKEN'))


def adfuller_test(ts):
    result = adfuller(ts)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for val, label in zip(result, labels):
        print(f"{label}: {val}")
    if result[1] <= 0.05:
        print("Data is stationary => timeseries rejects ADF null hypothesis")
    else:
        print("Data is non-stationary => timeseries does not reject null hypothesis")

def create_weekly_heatmap(ts, values_col='citation_count'):
    plt.figure(figsize=(12,12))
    sns.heatmap(ts.groupby([ts.index.hour,ts.index.weekday])[values_col].mean().unstack(), xticklabels = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'], linewidths=1)
    plt.title('[Hollywood] Average Citation Count',fontdict={'fontsize':22})
    plt.ylabel('Hour',fontdict={'fontsize':16})
    plt.xlabel('Day of Week',fontdict={'fontsize':16})