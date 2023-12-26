from scipy.stats import exponweib
import os
import pickle
import pandas as pd
import random
import numpy as np
#c is the shape parameter

keys = list()
keys.append('mu_s'),keys.append('mu_θ'),keys.append('mu_c'),keys.append('mu_∆c'),keys.append('mu_v_x'),keys.append('mu_v_y'),keys.append('mu_t_v'),keys.append('mu_t_a'),keys.append('mu_t_j'),
keys.append('mu_ω'),keys.append('std_s'),keys.append('std_θ'),keys.append('std_c'),keys.append('std_∆c'),keys.append('std_v_x'),keys.append('std_v_y'),keys.append('std_t_v'),keys.append('std_t_a'),
keys.append('std_t_j'),keys.append('std_ω'),keys.append('min_s'),keys.append('min_θ'),keys.append('min_c'),keys.append('min_∆c'),keys.append('min_v_x'),keys.append('min_v_y'),keys.append('min_t_v'),
keys.append('min_t_a'),keys.append('min_t_j'),keys.append('min_ω'),keys.append('max_s'),keys.append('max_θ'),keys.append('max_c'),keys.append('max_∆c'),keys.append('max_v_x'),keys.append('max_v_y'),
keys.append('max_t_v'),keys.append('max_t_a'),keys.append('max_t_j'),keys.append('max_ω'),keys.append('maxMinusMin_s'),keys.append('maxMinusMin_θ'),keys.append('maxMinusMin_c'),keys.append('maxMinusMin_∆c'),
keys.append('maxMinusMin_v_x'),keys.append('maxMinusMin_v_y'),keys.append('maxMinusMin_t_v'),keys.append('maxMinusMin_t_a'),keys.append('maxMinusMin_t_j'),keys.append('maxMinusMin_ω')    

def weibullProb(x, Scale, Shape):
	c = Shape
	# return np.log(c * np.power(x,c-1) * np.exp(-np.power(x,c)) )
	# return exponweib.cdf(x, a=1, c = Shape, loc = 0, scale = Scale)
	return np.log(exponweib.pdf(x, a=1, c = Shape, loc = 0, scale = Scale))
	# return exponweib.pdf(x, a=1, c = Shape, loc = 0, scale = Scale)


def weibullProb2(x, alpha, beta):
	return  np.log ((beta/alpha)* np.power(x/alpha, beta-1) * np.exp(-np.power(x/alpha, beta)))


os.chdir('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/weights/')
#Retrieving the saved weights from the weights folder        
with open('user9_dict', 'rb') as f:
    loaded_dict = pickle.load(f)

# print(loaded_dict)

with open('user9_dict_2', 'rb') as f:
    loaded_dict2 = pickle.load(f)

# print(loaded_dict2)
# print(weibullProb())

#Give the path of the FeatureVector.csv
#It will give you n random rows 
def getFeatures(featureVectorPath, n):
	df = pd.read_csv(featureVectorPath)
	# print(df.columns)
	indexes = random.sample(range(0,df.shape[0]), n)
	features = list()
	for i in indexes:
		features.append(df.iloc[i])
	return features

#Accessing user7's 5 random feature vectors
features = getFeatures('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user9 featureVector/FeatureVector.csv', 20)

f = features[0] 


indexes = list()
indexes.append(keys.index('max_v_x'))
indexes.append(keys.index('min_v_x'))
indexes.append(keys.index('maxMinusMin_v_x'))


# for i in indexes:
# 	print(weibullProb(f[i], loaded_dict[keys[i]][0], loaded_dict[keys[i]][1]))

# print('--------')

# for i in indexes:
# 	print(weibullProb2(f[i], loaded_dict2[keys[i]][0], loaded_dict2[keys[i]][1]))






# j = keys.index('maxMinusMin_v_x')
# print(keys[j])
# probs = list()
# for i in range(0,19):
	
# 	a = weibullProb(features[i][j], loaded_dict[keys[j]][0], loaded_dict[keys[j]][1])
# 	probs.append(a)
# 	print(a)

# print('-------')
# print(np.mean(probs))

def getProb(featureRow):
	prob = 0
	indexes = range(0,50)
	for i in indexes:
		prob += weibullProb(featureRow[i], loaded_dict[keys[i]][0], loaded_dict[keys[i]][1])

	return prob

def getProb2(featureRow):
	prob = 0
	indexes = range(0,50)
	for i in indexes:
		prob += weibullProb2(featureRow[i], loaded_dict[keys[i]][0], loaded_dict[keys[i]][1])

	return prob



df = pd.read_csv('/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user7 featureVector/FeatureVector.csv')
features = list()

#To make a prediction, we have to account for 10 consecutive strokes (as written in the research paper)
#Try for different strokes by using ranges like (2,11) , (3,12) , (4,13) , (5,14) 

for i in range(1,10): 
	features.append(df.iloc[i])

probs = list()
for i in range(0, 9): 
	probs.append(getProb2(features[i]))
	# print(getProb2(features[i]))

# print('-----------------')
print(np.mean(probs))









