import warnings
import numpy as np
import pandas as pd
# To fix numpy versioning error
np.float_ = np.float64
from prophet import Prophet
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-notebook")
from data.eda import get_timerange
from models.utils import train_test_split, print_rmse_mae

class FacebookProphet:
    def __init__(self, ts, split=0.8):
        self.ts = ts.reset_index().rename(columns={'issue_datetime': 'ds', 'citation_count': 'y'})
        self.train, self.test, self.split_date = train_test_split(self.ts, split)

    def plot_diagnostics(self):
        return self.results.plot_diagnostics(figsize=(10, 10))
    
    def predict(self):
        self.fore = self.model.predict(self.model.make_future_dataframe(periods=self.test.shape[0], freq='H'))
    
    def plot(self):
        steps = self.test.shape[0]//24
        train_ts = self.train.iloc[-steps:]
        test_ts = self.test.iloc[:steps]
        fore_ts = self.fore[(self.split_date - steps):(self.split_date + steps)]

        plt.figure(figsize=(20, 5))
        plt.plot(train_ts.ds, train_ts.y, label='Train')
        plt.plot(test_ts.ds, test_ts.y, label='Test')
        plt.plot(fore_ts.ds, fore_ts.yhat, label='Forecast')
        plt.fill_between(fore_ts.ds, fore_ts.yhat_lower, fore_ts.yhat_upper, alpha=0.5, label='Confidence')
        plt.title('Facebook Prophet Forecast', fontdict={'fontsize':20})
        plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig('./images/prophet_forecast.png')
        plt.show()
    
    def fit(self):
        self.model = Prophet(growth='flat',daily_seasonality=True,weekly_seasonality=True)
        self.model.fit(self.train)

    def measure(self):
        steps = self.test.shape[0]//24
        print_rmse_mae(
            self.train.iloc[-steps:, 1], 
            self.fore[(self.split_date - steps):self.split_date].yhat, 
            self.test.iloc[:steps, 1], 
            self.fore[self.split_date:(self.split_date + steps)].yhat
        )