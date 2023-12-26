import pandas as pd
import numpy as np
import os

#Check why are you getting infinity for mean and standard deviation for c and ∆c
#Maybe there is some mistake there

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# #Get dataframes of strokes of one user
# path = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/df1.csv'
# Df = pd.read_csv(path,index_col=False)
# # Df = Df.reset_index()
# # print(df.columns)
# # print(df.head())

# # Df = Df.iloc[0:1000]
# # print(Df)


# #The definition of stroke is the data points between two consecutive clicks. 
# #So the 'button' column is used for that.
# strokeList = list() #list of strokes

# indexes = Df.index[(Df['button'] != 'NoButton') & (Df['state'] != 'Move')].tolist()

# i = 0
# for j in indexes:
# 	strokeList.append(Df.iloc[i:j+1])  
# 	i = j+1

# strokeList.append(Df.iloc[indexes[-1]:Df.shape[0]])





# # print(str(len(strokeList)))


#Create the dataframe of feature vectors (i.e. the means and standard deviations of those features)
def getMean(x):
	return np.mean(x)

def getSTD(x):
	return np.std(x)

def getMin(x):
	return np.min(x)

def getMax(x):
	return np.max(x)

def getMaxMinusMin(x):
	return np.max(x) - np.min(x)

def getFeatureList(s, θ, c, dc, v_x, v_y, t_v, t_a, t_j, ω):
	#You are only giving the means and standard deviations of your features
	#You have to do this:
	#For every feature: Get the minimum, maximum, mean, standard deviation and (maximum-minimum)
	#You were calculating only two things out of your features before, which was mean and standard deviation
	#Now you will calculate minimum, maximum, mean, standard deviation and (maximum-minimum)

	#So you had these different values of s and changing values of time t
	#You only had mean(s) and std(s) at the end from one stroke
	#Now you will have min(s), max(s), mean(s), std(s), max(s)-min(s) from every stroke

	#Now from a lot of strokes, your featureVector.csv will be created  
	#Now for every feature(column in this featureVector.csv), 
	#Fit a separate Weibull Distribution for every feature
	#Let's say you have 1000 strokes, then you will have 1000 values of each feature
	#You will use these 1000 numbers to get two parameters for your weibull distribution
	#For each feature, you have this fitted Weibull Distribution
	#You can use this fitted distribution to calculate probability on new values of this feature
	#To calculate the final probability, assume features are independent of each other
	#which means, just multiply all those probabilities to get the final probability

	#See how you can decide a threshold to detect anomalies
	#Also see what they are talking about skewness in the paper, that's still mysterious.

	#Add the feature jitter(it is not jerk) and number of pauses later too, they will help too
	mu_s   = getMean(s  )
	mu_θ   = getMean(θ  )
	mu_c   = getMean(c  )
	mu_dc  = getMean(dc )
	mu_v_x = getMean(v_x)
	mu_v_y = getMean(v_y)
	mu_t_v = getMean(t_v)
	mu_t_a = getMean(t_a)
	mu_t_j = getMean(t_j)
	mu_ω   = getMean(ω  )

	std_s   = getSTD(s  )
	std_θ   = getSTD(θ  )
	std_c   = getSTD(c  )
	std_dc  = getSTD(dc )
	std_v_x = getSTD(v_x)
	std_v_y = getSTD(v_y)
	std_t_v = getSTD(t_v)
	std_t_a = getSTD(t_a)
	std_t_j = getSTD(t_j)
	std_ω   = getSTD(ω  )

	min_s   = getMin(s  )
	min_θ   = getMin(θ  )
	min_c   = getMin(c  )
	min_dc  = getMin(dc )
	min_v_x = getMin(v_x)
	min_v_y = getMin(v_y)
	min_t_v = getMin(t_v)
	min_t_a = getMin(t_a)
	min_t_j = getMin(t_j)
	min_ω   = getMin(ω  )

	max_s   = getMax(s  )
	max_θ   = getMax(θ  )
	max_c   = getMax(c  )
	max_dc  = getMax(dc )
	max_v_x = getMax(v_x)
	max_v_y = getMax(v_y)
	max_t_v = getMax(t_v)
	max_t_a = getMax(t_a)
	max_t_j = getMax(t_j)
	max_ω   = getMax(ω  )

	maxMinusMin_s   = getMaxMinusMin(s  )
	maxMinusMin_θ   = getMaxMinusMin(θ  )
	maxMinusMin_c   = getMaxMinusMin(c  )
	maxMinusMin_dc  = getMaxMinusMin(dc )
	maxMinusMin_v_x = getMaxMinusMin(v_x)
	maxMinusMin_v_y = getMaxMinusMin(v_y)
	maxMinusMin_t_v = getMaxMinusMin(t_v)
	maxMinusMin_t_a = getMaxMinusMin(t_a)
	maxMinusMin_t_j = getMaxMinusMin(t_j)
	maxMinusMin_ω   = getMaxMinusMin(ω  )

	tt = [mu_s, mu_θ, mu_c, mu_dc , mu_v_x, mu_v_y, mu_t_v, mu_t_a, mu_t_j, mu_ω, 
	std_s, std_θ, std_c, std_dc, std_v_x, std_v_y, std_t_v, std_t_a, std_t_j, std_ω,
	min_s, min_θ, min_c, min_dc , min_v_x, min_v_y, min_t_v, min_t_a, min_t_j, min_ω,
	max_s, max_θ, max_c, max_dc , max_v_x, max_v_y, max_t_v, max_t_a, max_t_j, max_ω,
	maxMinusMin_s, maxMinusMin_θ, maxMinusMin_c, maxMinusMin_dc , maxMinusMin_v_x, maxMinusMin_v_y, maxMinusMin_t_v, maxMinusMin_t_a, maxMinusMin_t_j, maxMinusMin_ω]
	return tt



# #As part of data cleaning, strokes with less than 4 points (i.e. 4 rows) are removed from the strokeList
# indexes =  list()
# for i in range(0, len(strokeList)):
# 	if strokeList[i].shape[0] <= 4:
# 		indexes.append(i)


# i = 0 
# while i < len(strokeList):
# 	if i in indexes:
# 		strokeList.pop(i)
# 		indexes = [x - 1 for x in indexes]
# 	i+=1




# f = pd.DataFrame(columns = ['mu_s', 'mu_θ', 'mu_c', 'mu_∆c', 'mu_v_x', 'mu_v_y', 'mu_t_v', 'mu_t_a', 'mu_t_j', 'mu_ω', 'std_s', 'std_θ', 'std_c', 'std_∆c', 'std_v_x', 'std_v_y', 'std_t_v', 'std_t_a', 'std_t_j', 'std_ω'])
# i = 0
# for df in strokeList:
#     #Add the feature vector in the feature dataframe f
# 	featureList = getFeatureList(df['s'].tolist(), df['θ'].tolist(), df['c'].tolist(), df['∆c'].tolist(), df['v_x'].tolist(), df['v_y'].tolist(), df['t_v'].tolist(), df['t_a'].tolist(), df['t_j'].tolist(), df['ω'].tolist())
# 	f.at[i,'mu_s']   = featureList[0]
# 	f.at[i,'mu_θ']   = featureList[1]
# 	f.at[i,'mu_c']   = featureList[2]
# 	f.at[i,'mu_∆c']  = featureList[3]
# 	f.at[i,'mu_v_x'] = featureList[4]
# 	f.at[i,'mu_v_y'] = featureList[5]
# 	f.at[i,'mu_t_v'] = featureList[6]
# 	f.at[i,'mu_t_a'] = featureList[7]
# 	f.at[i,'mu_t_j'] = featureList[8]
# 	f.at[i,'mu_ω']   = featureList[9]

# 	f.at[i,'std_s']   = featureList[10]
# 	f.at[i,'std_θ']   = featureList[11]
# 	f.at[i,'std_c']   = featureList[12]
# 	f.at[i,'std_∆c']  = featureList[13]
# 	f.at[i,'std_v_x'] = featureList[14]
# 	f.at[i,'std_v_y'] = featureList[15]
# 	f.at[i,'std_t_v'] = featureList[16]
# 	f.at[i,'std_t_a'] = featureList[17]
# 	f.at[i,'std_t_j'] = featureList[18]
# 	f.at[i,'std_ω']   = featureList[19]

# 	f.at[i,'min_s']   = featureList[20]
# 	f.at[i,'min_θ']   = featureList[21]
# 	f.at[i,'min_c']   = featureList[22]
# 	f.at[i,'min_∆c']  = featureList[23]
# 	f.at[i,'min_v_x'] = featureList[24]
# 	f.at[i,'min_v_y'] = featureList[25]
# 	f.at[i,'min_t_v'] = featureList[26]
# 	f.at[i,'min_t_a'] = featureList[27]
# 	f.at[i,'min_t_j'] = featureList[28]
# 	f.at[i,'min_ω']   = featureList[29]

# 	f.at[i,'max_s']   = featureList[30]
# 	f.at[i,'max_θ']   = featureList[31]
# 	f.at[i,'max_c']   = featureList[32]
# 	f.at[i,'max_∆c']  = featureList[33]
# 	f.at[i,'max_v_x'] = featureList[34]
# 	f.at[i,'max_v_y'] = featureList[35]
# 	f.at[i,'max_t_v'] = featureList[36]
# 	f.at[i,'max_t_a'] = featureList[37]
# 	f.at[i,'max_t_j'] = featureList[38]
# 	f.at[i,'max_ω']   = featureList[39]

# 	f.at[i,'maxMinusMin_s']   = featureList[40]
# 	f.at[i,'maxMinusMin_θ']   = featureList[41]
# 	f.at[i,'maxMinusMin_c']   = featureList[42]
# 	f.at[i,'maxMinusMin_∆c']  = featureList[43]
# 	f.at[i,'maxMinusMin_v_x'] = featureList[44]
# 	f.at[i,'maxMinusMin_v_y'] = featureList[45]
# 	f.at[i,'maxMinusMin_t_v'] = featureList[46]
# 	f.at[i,'maxMinusMin_t_a'] = featureList[47]
# 	f.at[i,'maxMinusMin_t_j'] = featureList[48]
# 	f.at[i,'maxMinusMin_ω']   = featureList[49]


# 	i+=1

# #When we will perform data cleaning, which includes cutting off the rows where x,y doesn't change and other things, 
# #the infinities coming in the featureVector should disappear.

# #Even after doing all the data cleaning, there might be some rows where we get infinities
# #This is to remove the row (stroke) which has infinity
# f.replace([np.inf, -np.inf], np.nan, inplace=True)
# # Drop rows with NaN
# f.dropna(inplace=True)

# #We get the minimum value of a column
# #We subtract this value from every element in that column
# #We do this to every column
# #We are doing this so that we can fix the location parameter to zero while fitting weibull. 
# #Location parameter zero can be used when the elements have lower bound zero
# for column in f.columns:
# 	f[column]-=f[column].min()



# ppath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/featureVector.csv'
# f.to_csv(path_or_buf = ppath, index = False)



# #joining the feature vectors generated from strokes (from multiple dataframes)
# pd.concat([data1, data2], axis=0)



################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
userDataframesPath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user9 dataframes/'


featureVectorList = list()
dfs = os.listdir(userDataframesPath)
if '.DS_Store' in dfs:
	dfs.remove('.DS_Store')

for df in dfs:
	dfPath = userDataframesPath + df
	Df = pd.read_csv(dfPath,index_col=False)

	########################################################################################
	strokeList = list() #list of strokes
	indexes = Df.index[(Df['button'] != 'NoButton') & (Df['state'] != 'Move')].tolist()

	i = 0
	for j in indexes:
		strokeList.append(Df.iloc[i:j+1])  
		i = j+1

	strokeList.append(Df.iloc[indexes[-1]:Df.shape[0]])
	########################################################################################

	indexes =  list()
	for i in range(0, len(strokeList)):
		if strokeList[i].shape[0] <= 4:
			indexes.append(i)

	i = 0 
	while i < len(strokeList):
		if i in indexes:
			strokeList.pop(i)
			indexes = [x - 1 for x in indexes]
		i+=1
	########################################################################################

	f = pd.DataFrame(columns = ['mu_s', 'mu_θ', 'mu_c', 'mu_∆c', 'mu_v_x', 'mu_v_y', 'mu_t_v', 'mu_t_a', 'mu_t_j', 'mu_ω', 'std_s', 'std_θ', 'std_c', 'std_∆c', 'std_v_x', 'std_v_y', 'std_t_v', 'std_t_a', 'std_t_j', 'std_ω'])
	i = 0

	for df in strokeList:
		featureList = getFeatureList(df['s'].tolist(), df['θ'].tolist(), df['c'].tolist(), df['∆c'].tolist(), df['v_x'].tolist(), df['v_y'].tolist(), df['t_v'].tolist(), df['t_a'].tolist(), df['t_j'].tolist(), df['ω'].tolist())
		f.at[i,'mu_s']   = featureList[0]
		f.at[i,'mu_θ']   = featureList[1]
		f.at[i,'mu_c']   = featureList[2]
		f.at[i,'mu_∆c']  = featureList[3]
		f.at[i,'mu_v_x'] = featureList[4]
		f.at[i,'mu_v_y'] = featureList[5]
		f.at[i,'mu_t_v'] = featureList[6]
		f.at[i,'mu_t_a'] = featureList[7]
		f.at[i,'mu_t_j'] = featureList[8]
		f.at[i,'mu_ω']   = featureList[9]
		f.at[i,'std_s']   = featureList[10]
		f.at[i,'std_θ']   = featureList[11]
		f.at[i,'std_c']   = featureList[12]
		f.at[i,'std_∆c']  = featureList[13]
		f.at[i,'std_v_x'] = featureList[14]
		f.at[i,'std_v_y'] = featureList[15]
		f.at[i,'std_t_v'] = featureList[16]
		f.at[i,'std_t_a'] = featureList[17]
		f.at[i,'std_t_j'] = featureList[18]
		f.at[i,'std_ω']   = featureList[19]
		f.at[i,'min_s']   = featureList[20]
		f.at[i,'min_θ']   = featureList[21]
		f.at[i,'min_c']   = featureList[22]
		f.at[i,'min_∆c']  = featureList[23]
		f.at[i,'min_v_x'] = featureList[24]
		f.at[i,'min_v_y'] = featureList[25]
		f.at[i,'min_t_v'] = featureList[26]
		f.at[i,'min_t_a'] = featureList[27]
		f.at[i,'min_t_j'] = featureList[28]
		f.at[i,'min_ω']   = featureList[29]
		f.at[i,'max_s']   = featureList[30]
		f.at[i,'max_θ']   = featureList[31]
		f.at[i,'max_c']   = featureList[32]
		f.at[i,'max_∆c']  = featureList[33]
		f.at[i,'max_v_x'] = featureList[34]
		f.at[i,'max_v_y'] = featureList[35]
		f.at[i,'max_t_v'] = featureList[36]
		f.at[i,'max_t_a'] = featureList[37]
		f.at[i,'max_t_j'] = featureList[38]
		f.at[i,'max_ω']   = featureList[39]
		f.at[i,'maxMinusMin_s']   = featureList[40]
		f.at[i,'maxMinusMin_θ']   = featureList[41]
		f.at[i,'maxMinusMin_c']   = featureList[42]
		f.at[i,'maxMinusMin_∆c']  = featureList[43]
		f.at[i,'maxMinusMin_v_x'] = featureList[44]
		f.at[i,'maxMinusMin_v_y'] = featureList[45]
		f.at[i,'maxMinusMin_t_v'] = featureList[46]
		f.at[i,'maxMinusMin_t_a'] = featureList[47]
		f.at[i,'maxMinusMin_t_j'] = featureList[48]
		f.at[i,'maxMinusMin_ω']   = featureList[49]
		i+=1
	########################################################################################
	f.replace([np.inf, -np.inf], np.nan, inplace=True)
	f.dropna(inplace=True)
	########################################################################################
	for column in f.columns:
		f[column]-=f[column].min()
	########################################################################################
	featureVectorList.append(f)
	########################################################################################



# # ppath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/featureVectors/'

# # i = 0
# # for featureVector in featureVectorList:
# # 	i+=1
# # 	path = ppath + 'featureVector' + str(i) + '.csv'
# # 	featureVector.to_csv(path_or_buf = path, index = False)


	

# ppath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/featureVectors/'
# featureVectors = os.listdir(ppath)
# if '.DS_Store' in featureVectors:
# 	featureVectors.remove('.DS_Store')
# print(featureVectors)

# featureDfs = list()
# for featureVector in featureVectors:
# 	featureDfs.append(pd.read_csv(ppath + featureVector))



featureVector = pd.concat(featureVectorList, axis=0)
path = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user9 featureVector/FeatureVector.csv'
featureVector.to_csv(path_or_buf = path, index = False)


































