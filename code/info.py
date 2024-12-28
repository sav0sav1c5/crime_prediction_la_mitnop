
#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% dropping columns
#NOTE: none of the date fields had any description so I'm not sure what they represent
columns_to_keep = ['DR_NO'
                   ,'Date Rptd'#date of report? 
                   ,'DATE OCC' #date of occurence?
                   ,'TIME OCC' #time of crime occurrence military time as int64
                               #example: 1800 is 18:00
                   ,'AREA' #area ID
                   ,'LAT' #lattitude
                   ,'LON' #longitiude
                   ]

# print(df.dtypes)

# df.index = pd.to_datetime(df['TIME OCC'], format='$')


#%% functions
def SelectWhereLATIsNullOrLONIsNull(df):
    '''Returns only rows where where LAT or LON are NA'''
    df_lat_lon_missing_values = df.loc[ (df['LAT'] == df['LON'])  & (df['LAT'] == 0) ]
    return df_lat_lon_missing_values
    ##old:
    # df_lat_lon_missing_values = df[df['LAT'] == df['LON'] ]
    # df_lat_lon_missing_values = df_lat_lon_isNULL_mask[df_lat_lon_isNULL_mask['LAT'] == 0]
    # return df_lat_lon_missing_values 

def GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE OCC Exact'):
    '''Returns passed the dataframe with the new column reprsenting 
    the exact date time of the occured crime'''
    
    df_TIME_Hours =  (df['TIME OCC'].astype(int) / 100).astype(int)
    df_TIME_Minutes = ((df['TIME OCC'].astype(int) %60)).astype(int) 
    # print('minutes=',df_TIME_Minutes, df_TIME_Minutes.dtypes)
    # print('hours=',df_TIME_Hours, df_TIME_Hours.dtypes)
    
    df_TimeStamp = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %H:%M:%S AM') 
    # df_TimeDelta = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
    df_TimeDelta = pd.Series(pd.to_timedelta(df_TIME_Hours,unit='hours'))
    df_TimeDelta += pd.Series(pd.to_timedelta(df_TIME_Minutes,unit='minutes'))
    df_TimeDelta += df_TimeStamp 
    
    df[new_column_name] = df_TimeDelta
    return df

def GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC min'):
    '''Returns passed the dataframe with the new column reprsenting 
    the minutes of day when the crime occured'''
    # df_TIME_Hours =  (df['TIME OCC'].astype(int) / 100).astype(int)
    # df_TIME_Minutes = ((df['TIME OCC'].astype(int) %60)).astype(int) 
    # df[new_column_name] = df_TIME_Hours * 60 + df_TIME_Minutes 
    df[new_column_name] = (df['TIME OCC'].astype(int)%60).astype(int)
    df[new_column_name] +=  (df['TIME OCC']/100).astype(int)*60
    return df



#%% load data frame

df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')

#%% drop columns
df = df[columns_to_keep]


#%% drop NA values for longitude
df = df.drop(SelectWhereLATIsNullOrLONIsNull(df).index)
# #%% 
# print(SelectWhereLATIsNullOrLONIsNull(df))


#%% get needed columns
df = GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC min')
df = GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE OCC Exact')

#%% divide into train and test
train_df , test_df = train_test_split(df, test_size=0.2)
# return train_df , test_df


#%% testing
if __name__=='__main__':
    # print(df.head(10))
    test_df = df.loc[ (df['DATE OCC Exact'].dt.year == 2020)
                     & (df['DATE OCC Exact'].dt.month == 3)
                     ]
    test_df = test_df.sort_values('DATE OCC Exact')
    
    print(test_df['DATE OCC Exact'])
    # print(df[['TIME OCC min']])
    

    
    
    
#df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
# print(df.head(10))
# print(df.index)
# columns_to_keep = ['DR_NO'
#                    ,'Date Rptd'
#                    ,'DATE OCC'
#                    ,'TIME OCC' #time of crime occurrence
#                    ,'AREA' #area ID
#                    ,'LAT' #lattitude
#                    ,'LON' #longitiude
#                    ]
# df = df[columns_to_keep]


# test_df['LAT'].scatter()
# print(test_df.columns)
# plt.scatter(test_df.index, test_df['LAT'])
# plt.show()

# print(df.columns)
# print(df['TIME OCC'].head(5))

# start = dt.datetime.now()
# sim_df['date'] = start + sim_df['cum_days'].map(dt.timedelta)
