# example of parametric probability density estimation
from matplotlib import pyplot as plt
from numpy.random import normal
from numpy import mean
from numpy import std
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import weibull_min
import pandas as pd
fig, ax = plt.subplots(1, 1)

df = pd.read_csv('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/featureVector.csv')
means = df.iloc[:,0:10]
stds = df.iloc[:,10:20]


dist = getattr(stats, 'weibull_min')
parameters = dist.fit(means)
print(parameters)




