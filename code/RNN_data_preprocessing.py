#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np


#%% functions used for handling data

def GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE TIME OCC'):
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

def GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC minutes'):
    '''Returns passed the dataframe with the new column reprsenting 
    the minutes of day when the crime occured'''
    # df_TIME_Hours =  (df['TIME OCC'].astype(int) / 100).astype(int)
    # df_TIME_Minutes = ((df['TIME OCC'].astype(int) %60)).astype(int) 
    # df[new_column_name] = df_TIME_Hours * 60 + df_TIME_Minutes 
    df[new_column_name] = (df['TIME OCC'].astype(int)%60).astype(int)
    df[new_column_name] +=  (df['TIME OCC']/100).astype(int)*60
    return df

def GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes'):
    '''Returns passed the dataframe with the new column reprsenting 
    the minutes since epoch when the crime occured'''
    ## minutes in this day
    df[new_column_name] = (df['TIME OCC'].astype(int)%60).astype(int)
    df[new_column_name] +=  (df['TIME OCC']/100).astype(int)*60
    ## up to this day
    # print(df.dtypes)
    df[new_column_name] += (df['DATE OCC'].dt.year - 1970) * 525948
    df[new_column_name] += (df['DATE OCC'].dt.month - 1) * 43829
    df[new_column_name] += (df['DATE OCC'].dt.day - 1) * 1440
    return df

def GetTimeDifferenceInMinutes(df,sort_by_column_name='TOTAL TIME OCC minutes',new_column_name = 'TIME DIFFERENCE minutes'):
    '''Returns passed the dataframe with the new column reprsenting 
    the difference in minutes from the previous row(instance? you know what I mean)
    Sorts the dataframe by the 'sort_by_column_name' column'''
    if (new_column_name in df.columns):
        df = df.drop(columns=new_column_name)
    df = df.sort_values(by=sort_by_column_name)
    df.reset_index(inplace=True)
    time_differences = np.zeros(shape=(df.shape[0]),dtype=np.int64)
    for row_id in range(1,df.shape[0]):
        time_differences[row_id] = df[sort_by_column_name][row_id] - df[sort_by_column_name][row_id-1]
        
    df.insert(loc=df.shape[1], column=new_column_name, value=time_differences )
    # print(df[[sort_by_column_name,new_column_name]])
    return df




def _getDataframeWithMaskedInput(df, timepoints_per_day = 24, mask_column_name='AVAILABLE MASK',mask_column_dtype = np.int64, mask_value= 0):
    '''returns a copy of the dataframe with the new column representing whether to use this value
    1 = value is not a copy and can be freely used
    0 = value is a copy and should be skipped'''
    
    # df = df.drop(columns=['index']) ## TODO:remove this
    df = df.sort_values(by='DATE TIME OCC')
    df = df.reset_index(drop=True)
    print(df['DATE TIME OCC'][0])
    print(df['DATE TIME OCC'][df.shape[0]-1])
    # df = df[0:200]
    first_date = df['DATE TIME OCC'][0]
    last_date = df['DATE TIME OCC'][df.shape[0]-1]
        
    # print(first_date.dt.total_seconds(), last_date.dt.total_seconds())
    diff = last_date - first_date
    seconds_in_a_day = 86400
    df_lenght = int((diff.total_seconds() / seconds_in_a_day ) * timepoints_per_day )
    
    new_df = df.copy()
    # new_df = new_df.reset_index(drop=True) ## deos not drop the fucking index
    
    delta_time_step_seconds = pd.Timedelta(int(seconds_in_a_day/timepoints_per_day), "s")
    delta_time_step_seconds_div2 = pd.Timedelta(int( (seconds_in_a_day/timepoints_per_day)/2 ), "s")
    
    
    ## set the time stamps to be the closest to the modulated position
    last_timestamp = new_df['DATE TIME OCC'][0]  
    i = 0   
    while (i < new_df.shape[0] ): 
        while ((last_timestamp - delta_time_step_seconds_div2 <= new_df['DATE TIME OCC'][i] <= last_timestamp + delta_time_step_seconds_div2 )
               == False):
            last_timestamp += delta_time_step_seconds 
        
        new_df['DATE TIME OCC'][i] = last_timestamp 
        i += 1
    ##end i loop
    print('Time stamps set to closest modulated position')
    
    ## drop rows with same DATE TIME OCC
    index_ranges_to_drop = []
    i = 0
    j = 1
    while (i <  new_df.shape[0]):
        j = i+1
        while (j < new_df.shape[0]):
            if (new_df['DATE TIME OCC'][i] != new_df['DATE TIME OCC'][j] ):
                break
            j += 1
            ## end j loop
        if ( j - i > 1):
            index_ranges_to_drop.append( (i+1,min(j+1,new_df.shape[0])) )
            # new_df = new_df.drop( index= range( i+1,j+1) )
            # new_df = new_df.sort_index().reset_index(drop=True)
            i = j
            # break
        i += 1
    ##end i loop
    
    # print('tmp')
    ## remove the flagged indexes
    for index_range in index_ranges_to_drop:
        new_df = new_df.drop( index= range( index_range[0],index_range[1] ) )
    new_df = new_df.sort_index().reset_index(drop=True)
    
    # new_df = new_df.reset_index(drop=True) ## reset the index since we removed some rows
    print('Duplicate DATE TIME OCC columns dropped, new shape=',new_df.shape)
    
    ## set the mask and add duplicates to the dataframe to fill in the missing values
    value_is_available_mask = []
    last_timestamp = new_df['DATE TIME OCC'][0]
    # last_timestamp += delta_time_step_seconds 
    i = 1
    value_is_available_mask.append(1)

    # print(new_df)    
    while ( i < new_df.shape[0] ):
        if (i % 10 == 0):
            print('done:', i,' / ' ,new_df.shape[0])
            print('last_timestamp = ',last_timestamp )
        # while (new_df['DATE TIME OCC'][i] == new_df['DATE TIME OCC'][i-1]):
        #     i+=1
        #     value_is_available_mask.append(0)
            
        last_timestamp = new_df['DATE TIME OCC'][i-1]
        last_timestamp += delta_time_step_seconds 
        
        while(last_timestamp != new_df['DATE TIME OCC'][i] ):
            
            new_row = new_df.loc[ i-1 ].copy()
            ##ADDITION
            ##set the values to the mask value
            # for column in new_df.columns:
            #     new_row[column] = mask_value
            ##---------
                
            new_row['DATE TIME OCC'] = last_timestamp 
            new_row = mask_value ## sets all the values to 0
            new_df.loc[ float(i-0.1) ] = new_row 
            last_timestamp += delta_time_step_seconds 
            new_df = new_df.sort_index().reset_index(drop=True)
            value_is_available_mask.append(0)
            i+=1
        
        value_is_available_mask.append(1)
        i += 1
    ##end of 'i' loop
    new_df.insert(loc= new_df.shape[1],column= mask_column_name,value= value_is_available_mask)
    # print(new_df)
    return new_df

def GetDataFrameWithMask():
    import Eksplorativna_analiza 
    loaded_df = None
    try:
        loaded_df = pd.read_csv('masked_RNN_input_value0_masked_rows_24_timepoints.csv')
        loaded_df['DATE TIME OCC'] = pd.to_datetime(loaded_df['DATE TIME OCC']
                                                    ,format='%Y-%m-%d %H:%M:%S'
                                                    ,errors='coerce' ## we have some 0 fields in the DATE TIME OCC column since we mask the input
                                                    )
        return loaded_df
    except:
        ## the csv does not exist so you must create it
        loaded_df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()        
        # print("TODO: uncomment it")
        # drop columns
        # print(df.columns)
        # df = data
        # df = df.drop(columns=[ 'Date Rptd', 'DATE OCC', 'AREA NAME',
        #         'Rpt Dist No', 'Part 1-2', 'Crm Cd Desc', 'Vict Age',
        #         'Status', 'Status Desc', 'LOCATION', 'YEAR OCC',
        #         'MONTH OCC', 'DAY OCC', 'QUARTER OCC'])
        # print(df.columns) ##['DR_NO', 'AREA', 'Crm Cd', 'LAT', 'LON', 'HOUR OCC', 'DATE TIME OCC'] are left
        # filter
        # print(df.shape)
        loaded_df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
        loaded_df = GetExactTimeOfCrimeOccurrence(loaded_df, new_column_name = 'DATE TIME OCC')
        loaded_df = loaded_df.loc[ (2023 <= loaded_df['DATE TIME OCC'].dt.year) & (loaded_df['DATE TIME OCC'].dt.year <= 2024 )]
        # df = df.loc[ (df['DATE TIME OCC'].dt.year == 2024 ) & (df['DATE TIME OCC'].dt.month == 1 ) & (df['DATE TIME OCC'].dt.day <= 15 )]
        # df = df.loc[ (df['DATE TIME OCC'].dt.year == 2024 ) ]
        ##getting the data 
        # print(df.shape)
        # print("-------------------------")
        new_df = _getDataframeWithMaskedInput(loaded_df ,timepoints_per_day = 24)
        # print(new_df.shape)
        # # print("-------------------------")
        # print(new_df)
        # # print("-------------------------")
        # print(new_df.dtypes)
        # # print("-------------------------")
        new_df.to_csv('masked_RNN_input_value0_masked_rows_24_timepoints.csv',index=False)
        return new_df 


