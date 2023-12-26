import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reliability.Distributions import Weibull_Distribution
import seaborn as sns
import pickle
import os

ppath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user9 featureVector/FeatureVector.csv'
df = pd.read_csv(ppath)

#0 to 49
column = df.columns[0]
data = df[column]

# print('Column ,Shape, Scale')
# params = stats.exponweib.fit(data, floc=0, f0=1)
# shape = params[1]
# scale = params[3]
# loc = 0
# print(column, shape, scale)

# dict = {}
# Shape = list()
# Scale = list()
# for column in df.columns:
# 	# print(column)
# 	data = df[column]
# 	params = stats.exponweib.fit(data)#, floc=0, f0=1)
# 	shape = params[1]
# 	scale = params[3]
# 	Shape.append(params[1])
# 	Scale.append(params[3])
# 	dict[column] = scale, shape

# # print(dict['min_v_x'])

# #Save the weights in a dictionary in the weights folder
# os.chdir('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/weights/')
# with open('user7_dict_2', 'wb') as f:
#     pickle.dump(dict, f)
# #Retrieving the saved weights from the weights folder        
# with open('user7_dict_2', 'rb') as f:
#     loaded_dict = pickle.load(f)
	
# print(loaded_dict['min_v_x'])	


from reliability.Fitters import Fit_Weibull_2P
import matplotlib.pyplot as plt
# wb = Fit_Weibull_2P(failures=list(data))
# # plt.show()

# print(wb.alpha)
# print(wb.beta)


dict = {}
for column in df.columns:
	# print(column)
	data = df[column]
	wb = Fit_Weibull_2P(failures=list(data))
	alpha = wb.alpha
	beta = wb.beta

	dict[column] = alpha, beta


#Save the weights in a dictionary in the weights folder
os.chdir('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/weights/')
with open('user9_dict_2', 'wb') as f:
    pickle.dump(dict, f)
#Retrieving the saved weights from the weights folder        
with open('user9_dict_2', 'rb') as f:
    loaded_dict = pickle.load(f)






# #Plotting the fitted distribution
# i = 25
# shape = Shape[i]
# scale = Scale[i]
# dist = Weibull_Distribution(alpha=scale, beta=shape)  # this created the distribution object
# dist.PDF()  
# plt.show()

#Plotting the real data to see how it looks (it always looks similar to the fitted distribution, it's pretty amazing)
# sns.set_style('white')
# sns.set_context("paper", font_scale = 2)
# sns.displot(data=df, x=column, kind="hist", bins = 100, aspect = 1.5)
# plt.show()


# Here are the features we have:
# mu_s
# mu_θ
# mu_c
# mu_∆c
# mu_v_x
# mu_v_y
# mu_t_v
# mu_t_a
# mu_t_j
# mu_ω
# std_s
# std_θ
# std_c
# std_∆c
# std_v_x
# std_v_y
# std_t_v
# std_t_a
# std_t_j
# std_ω
# min_s
# min_θ
# min_c
# min_∆c
# min_v_x
# min_v_y
# min_t_v
# min_t_a
# min_t_j
# min_ω
# max_s
# max_θ
# max_c
# max_∆c
# max_v_x
# max_v_y
# max_t_v
# max_t_a
# max_t_j
# max_ω
# maxMinusMin_s
# maxMinusMin_θ
# maxMinusMin_c
# maxMinusMin_∆c
# maxMinusMin_v_x
# maxMinusMin_v_y
# maxMinusMin_t_v
# maxMinusMin_t_a
# maxMinusMin_t_j
# maxMinusMin_ω




