from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-notebook")


def get_timerange(ts, start, end=None):
    if end == None:
        return ts.loc[(ts.index > start)]
    if start == None:
        return ts.loc[(ts.index < end)]
    return ts.loc[(ts.index > start) & (ts.index < end)]

def timeseries_decompose(ts):
    decomposition = seasonal_decompose(ts)
    return decomposition.trend, decomposition.seasonal, decomposition.resid

def plot_citations_per_year(counts):
    plt.figure(figsize=(20,5))
    sns.barplot(x=list(counts.index.year), y=counts.values)
    plt.title(f'Number of Citations per Year (2014-2023)', fontdict={'fontsize':22})
    plt.ylabel("Number of Citations (Millions)",fontdict={'fontsize':16})
    plt.xlabel("Year", fontdict={'fontsize':16})
    plt.savefig(f'./images/citations_per_year.png')

def plot_seasonality_decomp(ts, trend, seasonal, residuals):
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.title('Seasonality Decomposition for 2023')
    plt.plot(ts,label='Original',color='blue')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend,label='Trend',color='green')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonal',color='orange')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residuals,label='Residuals',color='red')
    plt.legend(loc='upper left')