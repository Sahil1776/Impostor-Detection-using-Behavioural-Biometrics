import pandas as pd
import numpy as np
import os

##########################################################################################################
# ------------------------------------------------------------------------------------------------------ #
# Just give the Path of the Data in the Format of the File used below, and it will generate the features #
# ------------------------------------------------------------------------------------------------------ #
##########################################################################################################



#0.1 seconds is the time you have to wait, if 0.1 second has passed and position didn't change, user's finder is at rest
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################


userPath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/Code/Mouse Dynamics/Mouse-Dynamics-Challenge-master/training_files/user9/'
#This function takes the path of the user's directory which has all the session files
#It generates a list of dataframes
#One dataframe of each session file (later I will divide the dataset into strokes)
#Majority of the features mentioned in the "Behavioual Biometrics..." paper are computed inside this loop and added to the dataframe


def getDataframes(userPath):
	dfList = list()
	sessions = os.listdir(userPath)
	for u in sessions:
		df = pd.DataFrame(columns = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y'])
		print(userPath+u)
		file = open(userPath+u, 'r')
		line = file.readline()
		j = 0
		for i in file:
			words = i.split(',')
			j+=1
			df.at[j] = [str(words[0]),str(words[1]),str(words[2]),str(words[3]),float(words[4]),float(str(words[5])[:-1])]
			# if j == 10000:
			# 	break
		
		#Before we start generating different features, we will perform data cleaning
		#This will also help us eliminate the infinities we are getting while generating features

        ################################################################
		#Doing Timestamp Cleaning (removing instances with the same timestamp)
		indexes =  list()
		t = list(df.iloc[:,0]) #record_timestamp
		
		for i in range(1, df.shape[0]):
			if (float(t[i]) - float(t[i-1]) == 0):
				indexes.append(i)
		df.drop(indexes, axis = 0, inplace= True)
		df.reset_index(drop = True, inplace = True)

		#Removing Null Space Events (instances with the same coordinates)
		x = df.iloc[:,4]
		y = df.iloc[:,5]
		i = 0
		end = int(len(x))
		indexes = list()
		while i < end-1:
			if x[i] == x[i+1] and y[i] == y[i+1]:
				indexes.append(i)
			i+=1
		df.drop(indexes, axis = 0, inplace = True)
		df.reset_index(drop = True,inplace = True)

		
		################################################################

		#Applying Linear Space Interpolation
		#Applying Cubic Spline Smoothing Interpolation
		#Applying Uniform Spatial Resampling

		################################################################
		
		#Adding the data now
		x = df.iloc[:,4]
		y = df.iloc[:,5]
		
		df['s'] = 'default value'
		for j in range(0,df.shape[0]):
			if j ==0:
				#It has just started, we can't put zero because it causes issues in division. So we added 1, just for the sake of it
				df.at[j,'s'] = 1
			else:
				df.at[j,'s'] = np.sqrt(np.power(df.at[j,'x'] - df.at[j-1,'x'],2) + np.power(df.at[j,'y'] - df.at[j-1,'y'],2))

		dx = list()
		dy = list()
		for k in range(0,df.shape[0]):
			if k ==0:
				dx.append(1)
				dy.append(1)
			else:
				dx.append(df.at[k,'x'] - df.at[k-1,'x'])
				dy.append(df.at[k,'y'] - df.at[k-1,'y'])


		dθ = list()
		dθ.append(1)
		for i in range(1,df.shape[0]):

			if dx[i] != 0:
				a = np.arctan(dy[i]/dx[i])
			else:
				a = 3.14/2

			if dx[i-1] != 0:
				b = np.arctan(dy[i-1]/dx[i-1])
			else:
				b = 3.14/2

			x = a - b

			if x<=0:
				x+= 2*3.14

			dθ.append(x)


		df['θ'] = 'default value'
		for i in range(0,df.shape[0]):
			df.at[i,'θ'] = float(np.arctan(dy[1]/dx[1])) + float(np.sum(dθ[2:i+2]))


		ds = list()
		ds.append(1)
		for i in range(1,df.shape[0]):
			ds.append(df.at[i,'s']-df.at[i-1,'s'])



		#Something is getting wrong with c, a lot of infinities are coming!!!!

		df['c'] = 'default value'
		for i in range(0,df.shape[0]):
			if i ==0:
				df.at[i,'c'] = 1
			else:
				df.at[i,'c'] = dθ[i]/df.at[i,'s']

		dc = list()
		dc.append(1)
		for i in range(1,df.shape[0]):
			dc.append(df.at[i,'c']-df.at[i-1,'c'])

		df['∆c'] = 'default value'
		for i in range(0,df.shape[0]):
			if i ==0:
				df.at[i,'∆c'] = 1
			else:
				df.at[i,'∆c'] = dc[i]/df.at[i,'s']


		df['v_x'] = 'default value'
		df['v_y'] = 'default value'
		df['t_v'] = 'default value'
		df['t_a'] = 'default value'
		df['t_j'] = 'default value'
		df['ω'] = 'default value'

		dt = list()

		for i in range(0,df.shape[0]):
			if i == 0:
				diff = 1
			else:
				diff = float(df.at[i,'record_timestamp']) - float(df.at[i-1,'record_timestamp'])
			if diff == 0:
				diff = 1
			dt.append(diff)

		
		for i in range(0,df.shape[0]):
			if i==0:
				df.at[i,'v_x'] = 1
			else:
				df.at[i,'v_x'] = dx[i]/dt[i]

		for i in range(0,df.shape[0]):
			if i==0:
				df.at[i,'v_y'] = 1
			else:
				df.at[i,'v_y'] = dy[i]/dt[i]


		for i in range(0,df.shape[0]):
			df.at[i,'t_v'] = np.sqrt(df.at[i,'v_x']*df.at[i,'v_x'] + df.at[i,'v_y']*df.at[i,'v_y'])


		dv = list()
		dv.append(1)
		for i in range(1,df.shape[0]):
			dv.append(df.at[i,'t_v'] - df.at[i-1,'t_v'])


		for i in range(0,df.shape[0]):
			if i==0:
				df.at[i,'t_a'] = 1
			else:
				df.at[i,'t_a'] = dv[i]/dt[i]


		dt_a = list()
		dt_a.append(1)
		for i in range(0,df.shape[0]):
			if i ==0:
				diff = 1
			else:
				diff = float(df.at[i,'t_a']) - float(df.at[i-1,'t_a'])
			if diff == 0:
				diff = 1
			dt_a.append(diff)


		for i in range(0,df.shape[0]):
			if i==0:
				df.at[i,'t_j'] = 1
			else:
				df.at[i,'t_j'] = dt_a[i]/dt[i]


		dθt = list()
		dθt.append(1)
		for i in range(0,df.shape[0]):

			if dt[i] != 0:
				a = np.arctan(dy[i]/dt[i])
			else:
				a = 3.14/2

			if dt[i-1] != 0:
				b = np.arctan(dy[i-1]/dt[i-1])
			else:
				b = 3.14/2

			x = a - b
			if x<=0:
				x+= 2*3.14

			dθt.append(x)

		for i in range(0,df.shape[0]):
			df.at[i,'ω'] = float(np.arctan(dy[1]/dt[1])) + float(np.sum(dθt[2:i+2]))

		

		dfList.append(df)

		
	return dfList




dfList = getDataframes(userPath)

i = 0
for df in dfList:
	i+=1
	ppath = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/My Code/user9 dataframes/df' + str(i) + '.csv'
	df.to_csv(path_or_buf = ppath, index = False)


##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################


# path = '/Users/sahilkakkar/Downloads/Major Project (Impostor Detection)/Code/Mouse Dynamics/Mouse-Dynamics-Challenge-master/training_files/user9/session_3390119815'
# df = pd.DataFrame(columns = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y'])

# def accessData(path):
# 	file = open(path, 'r')
# 	line = file.readline()
# 	j = 0
# 	for i in file:
# 		words = i.split(',')
# 		j+=1
# 		df.at[j] = [str(words[0]),
# 		            str(words[1]),
# 		            str(words[2]),
# 		            str(words[3]),
# 		            float(words[4]),
# 		            float(str(words[5])[:-1])]
# 		if j==10:
# 			break

# accessData(path)


# #Data Cleaning

# #Stroke = Set of points between 2 mouse clicks
# #Removing the data points which are almost at the same timestamp
# #Ignoring patterns with less than 4 points
# #Removing null space events


# #Smoothing

# #Feature Extraction
# #I will extract all the features
# #In Anomaly Detection, learning time is not that much
# #You just have to calculate the mean and standard deviation of the features
# #and give it to compute p(x)
# #The real thing to learn is the features you use
# #Some features will help greatly, some will be useless
# #That's what I have to see.

# #After smoothing the x and y coordinates of one stroke,
# #I obtain these new features:

# #horizontal coordinates x' (x after smoothing)
# #vertical coordinates y' (y after smoothing)
# #path distance from the origin s' (s after smoothing)
# #angle of the path tangent with the x-axis θ
# #curvature c
# #derivative of curvature ∆c
# x = df.iloc[:,4]
# y = df.iloc[:,5]

# df['s'] = 'default value'
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		#It has just started, we can't put zero because it causes issues in division. So we added 1, just for the sake of it
# 		#This 1 will equally affect every user, so shouldn't be a problem
# 		df.at[i,'s'] = 1
# 	else:
# 		df.at[i,'s'] = np.sqrt(df.at[i,'x']*df.at[i,'x'] + df.at[i,'y']*df.at[i,'y'])


# dx = list()
# dy = list()
# dx.append(0)
# dy.append(0)
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		dx.append(1)
# 		dy.append(1)
# 	else:
# 		dx.append(df.at[i,'x'] - df.at[i-1,'x'])
# 		dy.append(df.at[i,'y'] - df.at[i-1,'y'])



# dθ = list()
# dθ.append(1)
# for i in range(1,i+1):

# 	if dx[i] != 0:
# 		a = np.arctan(dy[i]/dx[i])
# 	else:
# 		a = 3.14/2

# 	if dx[i-1] != 0:
# 		b = np.arctan(dy[i-1]/dx[i-1])
# 	else:
# 		b = 3.14/2
# 	x = a - b

# 	if x<=0:
# 		x+= 2*3.14
# 	dθ.append(x)


# df['θ'] = 'default value'
# for i in range(1,df.shape[0]+1):
# 	df.at[i,'θ'] = float(np.arctan(dy[1]/dx[1])) + float(np.sum(dθ[2:i+2]))


# ds = list()
# ds.append(1)
# for i in range(1,df.shape[0]+1):
# 	if i == 1:
# 		ds.append(df.at[i,'s'])
# 	else:
# 		ds.append(df.at[i,'s']-df.at[i-1,'s'])


# df['c'] = 'default value'
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		df.at[i,'c'] = 1
# 	else:
# 		df.at[i,'c'] = dθ[i]/ds[i]

# dc = list()
# dc.append(1)
# for i in range(1,df.shape[0]+1):
# 	if i == 1:
# 		dc.append(df.at[i,'c'])
# 	else:
# 		dc.append(df.at[i,'c']-df.at[i-1,'c'])


# df['∆c'] = 'default value'
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		df.at[i,'∆c'] = 1
# 	else:
# 		df.at[i,'∆c'] = dc[i]/ds[i]

# #--------------------------------------------------------------------------------

# #9 more features
# # horizontal coordinates x (without smoothing)
# # vertical coordinates y (without smoothing)
# # t (timestamp)
# # horizontal velocity v_x
# # vertical velocity v_y
# # tangential velocity t_v
# # tangential acceleration t_a
# # tangential jerk t_j
# # angular velocity ω

# df['v_x'] = 'default value'
# df['v_y'] = 'default value'
# df['t_v'] = 'default value'
# df['t_a'] = 'default value'
# df['t_j'] = 'default value'
# df['ω'] = 'default value'

# dt = list()
# dt.append(1)
# for i in range(1,df.shape[0]+1):
# 	if i == 1:
# 		diff = 1
# 	else:
# 		diff = float(df.at[i,'client_timestamp']) - float(df.at[i-1,'client_timestamp'])
# 	if diff == 0:
# 		diff = 1
# 	dt.append(diff)

# for i in range(1,df.shape[0]+1):
# 	if i==1:
# 		df.at[i,'v_x'] = 1
# 	else:
# 		df.at[i,'v_x'] = dx[i]/dt[i]


# for i in range(1,df.shape[0]+1):
# 	if i==1:
# 		df.at[i,'v_y'] = 1
# 	else:
# 		df.at[i,'v_y'] = dy[i]/dt[i]


# for i in range(1,df.shape[0]+1):
# 	df.at[i,'t_v'] = np.sqrt(df.at[i,'v_x']*df.at[i,'v_x'] + df.at[i,'v_y']*df.at[i,'v_y'])
	

# dv = list()
# dv.append(1)
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		dv.append(1)
# 	else:
# 		dv.append(df.at[i,'t_v'] - df.at[i-1,'t_v'])



# for i in range(1,df.shape[0]+1):
# 	if i==1:
# 		df.at[i,'t_a'] = 1
# 	else:
# 		df.at[i,'t_a'] = dv[i]/dt[i]


# dt_a = list()
# dt_a.append(1)
# for i in range(1,df.shape[0]+1):
# 	if i ==1:
# 		diff = 1
# 	else:
# 		diff = float(df.at[i,'t_a']) - float(df.at[i-1,'t_a'])
# 	if diff == 0:
# 		diff = 1
# 	dt_a.append(diff)

# for i in range(1,df.shape[0]+1):
# 	if i==1:
# 		df.at[i,'t_j'] = 1
# 	else:
# 		df.at[i,'t_j'] = dt_a[i]/dt[i]







# dθt = list()
# dθt.append(1)
# for i in range(1,i+1):

# 	if dt[i] != 0:
# 		a = np.arctan(dy[i]/dt[i])
# 	else:
# 		a = 3.14/2

# 	if dt[i-1] != 0:
# 		b = np.arctan(dy[i-1]/dt[i-1])
# 	else:
# 		b = 3.14/2
# 	x = a - b

# 	if x<=0:
# 		x+= 2*3.14
# 	dθt.append(x)



# for i in range(1,df.shape[0]+1):
# 	df.at[i,'ω'] = float(np.arctan(dy[1]/dt[1])) + float(np.sum(dθt[2:i+2]))



# #Add these features later too, they might help:

# # Almost all features were used (58 from the total of 63) and the vector of features has different lengths
# # for the different users, ranging from 1 to 11 features. The average size of the features vector is 5.
# # The 5 most frequently used features by the several users are: max(vy) (used by 21 users); min(v) (16 users);
# # max(vx) − min(vx) (12 users); paused time (11 users); jitter (11 user). In figure 9 we present the histogram
# # of the feature vector sizes for all the users.


		
#Before we start generating different features, we will perform data cleaning
#This will also help us eliminate the infinities we are getting while generating features


################################################################















