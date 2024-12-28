#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *


#%% class declaration

class SimpleMaskedRNN:
    def __init__(self, input_shape, output_len, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = keras.models.Sequential()
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.model.add(keras.layers.Masking(mask_value=0,input_shape=self.input_shape) )
        self.model.add(keras.layers.GRU(name='GRU_layer'
                                        ,units=10 ##output shape of this layer
                                        # , input_shape=self.input_shape
                                        ))
        self.model.add(keras.layers.Dense(self.output_len))
    
    def _compileModel(self):
        self.model.compile(loss=self.loss
                           , optimizer="adam"
                           , metrics=[keras.metrics.MeanSquaredError()])
    
        

#%% function definition
def GetSimpleMaskedRNNWithTrainingHistory(df = None,epochs=3, timeseries_batch_size= 10, timepoints_per_day= 24 ):
    ''' returns the 'SimpleRNN' object with its training history 
    don't change timeseries_batch_size= 10, timepoints_per_day= 24 since I hard coded it to read from a csv file so I don't have to re-create it each time I make a change to the model
    '''

    
    if df == None:
        # df = pd.read_csv('masked_RNN_input_value0_masked_rows_24_timepoints.csv')
        # df['DATE TIME OCC'] = pd.to_datetime(df['DATE TIME OCC']
        #                                        ,format='%Y-%m-%d %H:%M:%S'
        #                                       ,errors='coerce'
        #                                       )
        df = GetDataFrameWithMask()
        # print(df)
        df.loc[df['AVAILABLE MASK'] == 0] = 0 ## set the rows with missing values to all be equal to 0 (0 is the set mask value in the neural network class)
        # print(df)
    

    ## %% train and validation split
    train_to_validation_split = 0.9
    train_df = df
    
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    
    ##%% create time series
    ## these 2 paramaters are now function paramaters
    timepoints_per_day = 24 ## overwritting it because I forgot that I should not overwrite it
    timeseries_sequence_length = timepoints_per_day 
    timeseries_batch_size = timepoints_per_day
    
    input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    target_columns = ['LAT','LON']
    
    train_data = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= train_df[input_columns],
        targets= train_df[target_columns],
        sequence_length= timeseries_sequence_length,
        batch_size= timeseries_batch_size 
        )
    
    validation_data = tf.keras.preprocessing.timeseries_dataset_from_array(
        data= validation_df[input_columns],
        targets= validation_df[target_columns],
        sequence_length= timeseries_sequence_length,
        batch_size= timeseries_batch_size 
        )
    
    # print(train_data)
    # for batch in train_data:
    #     inputs, targets = batch
    #     print('inputs=',inputs)
    #     print('targets=',targets )
    # print(train_data )
    
    
    
    ##%% create and compile
    rnn_masked = SimpleMaskedRNN(input_shape= (timeseries_sequence_length,len(input_columns)) 
                          ,output_len=len(target_columns )  
                          )
    print(rnn_masked .model.summary())
    
    ##%% train
    training_history = rnn_masked.model.fit(train_data
                                            , epochs= epochs
                                            , shuffle= False
                                            , validation_data= validation_data )
        
    return rnn_masked ,training_history 
        
    
#%% main function
if __name__ == '__main__':
    # import Eksplorativna_analiza
    # data = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    rnn_masked, training_hitstory = GetSimpleMaskedRNNWithTrainingHistory()
