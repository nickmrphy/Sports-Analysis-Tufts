#!/usr/bin/env python
# coding: utf-8

# https://harvardsportsanalysis.wordpress.com/2011/11/30/how-to-value-nfl-draft-picks/
# 
# http://nbasense.com/draft-pick-trade-value/trade-simulator

# In[1]:
# Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

# In[2]:
# Get the draft picks to give/receive from the user
# You can assume that this input will be entered as expected
# DO NOT CHANGE THESE PROMPTS
print("\nSelect the picks to be traded away and the picks to be received in return.")
print("For each entry, provide 1 or more pick numbers from 1-60 as a comma-separated list.")
print("As an example, to trade the 1st, 3rd, and 25th pick you would enter: 1, 3, 25.\n")
give_str = input("Picks to give away: ")
receive_str = input("Picks to receive: ")

# Convert user input to an array of ints
give_picks = list(map(int, give_str.split(',')))
receive_picks = list(map(int, receive_str.split(',')))


# Read in data
draft_picks = pd.read_csv('draftDB.csv')
players = pd.read_csv('playerDB.csv')


# In[3]:


# Function to take in the basketball reference url string
# and output the player ID
def getID(url_string):
    # default to NaN so we can drop records without an ID
    final_id = math.nan
    try:
        # Index all characters after the last backslash
        url_string = url_string.rsplit('/',1)[1]
        # Index all characters expect the last 5 (removes '.html')
        final_id = url_string[:-5]
    except:
        pass
    return(final_id)


# In[4]:


# Convert data types
draft_picks['basketball_reference_url'] = draft_picks['basketball_reference_url'].astype(str)
players['Season'] = players['Season'].astype(str)
draft_picks['yearDraft'] = draft_picks['yearDraft'].astype(int)


# In[5]:


# Apply function to get player IDs and drop any NaNs (records without IDs)
draft_picks['urlID'] = draft_picks['basketball_reference_url'].apply(lambda x: getID(x))


# In[6]:

# In[7]:


# Only use players drafted between 1979 and 2015. This will ensure we should have 4 years of data on them
draft_picks = draft_picks[(draft_picks.yearDraft >= 1979) & (draft_picks.yearDraft <= 2015)].reset_index(drop = True)

# Drop any players drafted after the 60th pick for consistency between draft years
draft_picks = draft_picks[(draft_picks.numberPickOverall <= 60)].reset_index(drop = True)


# In[8]:


# In[9]:


# Subset each players draft year
draft_years = draft_picks.loc[:,['urlID','yearDraft']].copy()

# Join the draft years in and drop and players which we do not have draft data on
players = players.merge(draft_years, how = 'left', on = 'urlID')
players = players.dropna(0,how = 'any',subset=['yearDraft']).reset_index(drop = True)


# In[10]:


# Create a season start year column to calculate draft year offset
players['season_start_yr'] = players['Season'].apply(lambda x: int(x[0:4]))

# Create draft year offset to measure how many seasons it has been since player was drafted
players['draft_year_offset'] = players['season_start_yr'] - players['yearDraft']

# Only use players drafted between 1979 and 2015. This will ensure we should have 4 years of data on them
players = players[(players.yearDraft >= 1979) & (players.yearDraft <= 2015)].reset_index(drop = True)


# In[11]:


# Merge draft data and season stats together
df = draft_picks.merge(players, how = 'left', on = ['urlID'])

# Drop any season stats after the player's 4th season
df = df[(df['draft_year_offset'] <= 3) | (pd.isnull(df).any(1))]


# In[12]:


# Drop unncessary columns
cols_to_drop = ['numberRound','numberRoundPick','slugTeam','idPlayer','idTeam','basketball_reference_url'
               ,'Rk','Player','Pos','Tm','G', 'MP', 'PER', 'TS%', '3PAr', 'FTr','ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS'
               ,'DWS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP','yearDraft_y','Season']
df = df.drop(cols_to_drop, axis = 1)


# In[13]:


# In[14]:


# Calculate aggregates on each player: Total Win Shares, Age when Drafted, International Flag
grouped = df.groupby(['namePlayer', 'numberPickOverall']).agg({'WS': ['sum'], 'Age': ['min']}).reset_index(drop = False)
grouped.columns = ['namePlayer', 'numberPickOverall', 'total_WS', 'draft_age']
grouped = grouped.reset_index(drop = True)


# In[15]:


# Calculate the average total WS by draft pick
wsoav = grouped.groupby(['numberPickOverall']).agg({'total_WS': ['mean']}).reset_index(drop = False)
wsoav.columns = ['numberPickOverall', 'avg_WS']
wsoav = wsoav.reset_index(drop = True)


# In[16]:


# In[17]:


X_pick = wsoav['numberPickOverall'].to_numpy().reshape(-1,1)
Y_pick = wsoav['avg_WS'].to_numpy().reshape(-1,1)


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X_pick, Y_pick, test_size=0.2, random_state=58)


# In[19]:

# In[20]:


# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=500000
degree = 3
model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
            normalize=True,cv=5))
model.fit(X_train,Y_train.ravel())
test_pred = np.array(model.predict(X_test))
RMSE=np.sqrt(np.sum(np.square(test_pred-Y_test)))
test_score = model.score(X_test,Y_test)

# In[21]:

all_picks = np.linspace(1,60,60).reshape(-1,1)
all_picks_pred = np.array(model.predict(all_picks))

# In[22]:

'''
x = range(61)
y = range(-1,5)
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

ax1.scatter(X_test, Y_test, c='blue', label='Test data')
ax1.scatter(X_train, Y_train, c='orange', label='Train data')
#plt.xlabel('Draft Pick #')
#plt.ylabel('WS Compared to Avg. Pick')
plt.plot(all_picks, all_picks_pred, color='black', linewidth=2, label = 'Model Prediction')
plt.legend(loc='upper right');
plt.show()
'''

# In[23]:


# In[24]:


def trade_eval(rec,out):
    rec_val = model.predict(np.array(rec).reshape(-1,1)).sum()
    out_val = model.predict(np.array(out).reshape(-1,1)).sum()
    if rec_val > out_val:
        return True
    elif rec_val < out_val:
        return False
    else:
        print("\nTrade result: This trade returns equal value. Consult with the GM.\n")


# In[35]:



# Print feeback on trade
# DO NOT CHANGE THESE OUTPUT MESSAGES
if trade_eval(receive_picks, give_picks) == True:
    print("\nTrade result: Success! This trade receives more value than it gives away.\n")
    print("Based on our model, the total value you would receive is: {}".format(str(model.predict(np.array(receive_picks).reshape(-1,1)).sum())))
    print("And the total value you would give away is: {}".format(str(model.predict(np.array(give_picks).reshape(-1,1)).sum())))
else:
    print("\nTrade result: Don't do it! This trade gives away more value than it receives.\n")
    print("Based on our model, the total value you would receive is: " + str(model.predict(np.array(receive_picks).reshape(-1,1)).sum()))
    print("And the total value you would give away is: " + str(model.predict(np.array(give_picks).reshape(-1,1)).sum()))


# In[ ]:




