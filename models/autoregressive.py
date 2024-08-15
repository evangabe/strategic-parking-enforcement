import warnings
import pandas as pd
import itertools
import statsmodels.api as sm
from data.eda import get_timerange
import matplotlib.pyplot as plt

from models.utils import train_test_split, print_rmse_mae

plt.style.use("seaborn-v0_8-notebook")
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)

class Autoregressive:
    def __init__(self, ts, split=0.8):
        self.ts = ts
        self.train, self.test, self.split_date = train_test_split(ts, split)

    def plot_diagnostics(self):
        return self.results.plot_diagnostics(figsize=(10, 10))
    
    def predict(self, dynamic=True):
        steps = self.test.shape[0]//24
        start = self.train.index[-steps]
        self.pred = self.results.get_prediction(start=start, dynamic=dynamic, full_results=True)
        self.fore = self.results.get_forecast(steps=steps)
    
    def plot(self, type='ARIMA'):
        pred_conf, fore_conf = self.pred.conf_int(), self.fore.conf_int()

        plt.figure(figsize=(20,6))
        ax = get_timerange(self.ts, pred_conf.index[0], fore_conf.index[-1]).plot(label='Actual')
        ax.fill_between(pred_conf.index, pred_conf.iloc[:,0], pred_conf.iloc[:,1], alpha=0.5, color="orange", label='confidence')
        ax.fill_between(fore_conf.index, fore_conf.iloc[:,0], fore_conf.iloc[:,1], alpha=0.5, color="orange")
        self.pred.predicted_mean.plot(ax=ax, label='Prediction', alpha=0.9)
        self.fore.predicted_mean.plot(ax=ax, label='Forecast', alpha=0.9)
        ax.set_ylim(bottom=0)
        plt.title(f'{type} Forecast', fontdict={'fontsize':20})
        plt.ylabel('Citation Count')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig(f'./images/{type.lower()}_forecast.png')
        plt.show()

    def _get_model(self, type='ARIMA', **kwargs):
        if type == 'ARIMA':
            return sm.tsa.ARIMA(self.train, enforce_invertibility=False, enforce_stationarity=False, **kwargs)
        elif type == 'SARIMAX':
            return sm.tsa.SARIMAX(self.train, enforce_invertibility=False, enforce_stationarity=False, **kwargs)
        else:
            print("Model not defined properly. Please use ARIMA / SARIMAX helper classes.")
    
    def fit(self, type, **kwargs):
        model = self._get_model(type, **kwargs)
        self.results = model.fit()

    def measure(self):
        steps = self.test.shape[0]//24
        print_rmse_mae(
            self.train.iloc[-steps:], 
            self.pred.predicted_mean, 
            self.test.iloc[:steps], 
            self.fore.predicted_mean[self.split_date:]
        )

class ARIMA(Autoregressive):
    def __init__(self, ts, split=0.8):
        super().__init__(ts, split)
        self.type = 'ARIMA'
    
    def fit(self, order):
        super().fit(self.type, **{"order": order})

    def plot(self):
        super().plot(self.type)

    def predict(self):
        super().predict()

class SARIMAX(Autoregressive):

    def __init__(self, ts, split=0.8):
        super().__init__(ts, split)
        self.type = 'SARIMAX'
    
    def fit(self, order, s_order):
        super().fit(type=self.type, kwargs={ "order": order, "seasonal_order": s_order })
    
    def plot(self):
        super().plot(self.type)

    def predict(self):
        super().predict(dynamic=False)

    def grid_search(self, max_value=2, display=True):

        # Define the p, d, q parameters to take any value between 0 and 2
        p = d = q = range(0, max_value)

        # Generate all different combinations of p, d, and q
        pdq = list(itertools.product(p, d, q))

        # Basis represents number of periods in a "season" (7*24 = 168 weekly data points)
        basis = 24
        seasonal_pdq = [(val[0], val[1], val[2], basis) for val in pdq]

        # List to store results
        results = []

        # Grid search for SARIMAX parameters
        for order in pdq:
            for seasonal_order in seasonal_pdq:
                try:
                    # SARIMAX Model
                    model = sm.tsa.SARIMAX(self.ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                    
                    # Fit model with training constraints (for speed)
                    model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
                    
                    # Store results in a dictionary
                    results.append({
                        'p': order[0],
                        'd': order[1],
                        'q': order[2],
                        'P': seasonal_order[0],
                        'D': seasonal_order[1],
                        'Q': seasonal_order[2],
                        's': basis,
                        'AIC': model_fit.aic
                    })
                    
                    print(f'{self.type}{order} w/ {seasonal_order} seasonality - AIC: {model_fit.aic}')
                    
                except Exception as e:
                    print(f'{self.type}{order} failed with error: {e}')
                    continue
        
        self.grid_search_results = pd.DataFrame(results).sort_values(by='AIC', ascending=True).reset_index(drop=True)

        if display:
            print(self.grid_search_results.head(10))

    def get_best_params(self):
        best_order = list(self.grid_search_results.iloc[0, :3])
        best_seasonal_order = list(self.grid_search_results.iloc[0, 3:-1])
        best_aic = self.grid_search_results.iloc[0, -1]
        return best_order, best_seasonal_order, best_aic