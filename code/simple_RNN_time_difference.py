#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *

#%%

class SimpleRNN:
    def __init__(self, input_shape, output_len, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = keras.models.Sequential()
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.model.add(keras.layers.GRU(name='GRU_layer',
            units=10, ##output shape of this layer
            input_shape=self.input_shape
            ))
        self.model.add(keras.layers.Dense(self.output_len))
    
    def _compileModel(self):
        self.model.compile(loss=self.loss
                           , optimizer="adam"
                           , metrics=[keras.metrics.MeanSquaredError()])



#%% function definition
def GetSimpleRNNWithTrainingHistory(df=None,epochs=3, timeseries_batch_size= 10, timeseries_sequence_length= 5 ):
    ''' returns the  'SimpleRNN' object with its training history 
    'timeseries_sequence_length' can be set to practicaly anything since we use the time difference in minutes between instances'''


    ##%% preprocessing 
    if (df == None):
        import Eksplorativna_analiza
        df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
        
    df = GetExactTimeOfCrimeOccurrence(df,new_column_name = 'DATE TIME OCC')
    df = GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC minutes')
    df = GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes')
    df = GetTimeDifferenceInMinutes(df,new_column_name = 'TIME DIFFERENCE minutes')
    # print(df[['TOTAL TIME OCC minutes','TIME DIFFERENCE minutes']])
    # print(df[['TIME OCC', 'DATE OCC', 'DATE TIME OCC']])
    # print(df.dtypes)
    
    ##%% train and validation split
    train_to_validation_split = 0.9
    train_df = df.loc[ (df['DATE OCC'].dt.year >= 2023) 
                      & (df['DATE OCC'].dt.year <= 2024) ]
    
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    
    ##%% create time series
    ## these 2 paramaters are now function paramaters
    # timeseries_sequence_length = 5
    # timeseries_batch_size = 10
    
    input_columns = ['TIME DIFFERENCE minutes','YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
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
    # del rnn
    rnn = SimpleRNN(input_shape= (timeseries_sequence_length,len(input_columns)) 
                    ,output_len=len(target_columns )  
                    )
    print(rnn.model.summary())
    
    ##%% train
    training_history = rnn.model.fit(train_data
                                     , epochs= epochs
                                     , shuffle= False
                                     , validation_data= validation_data )
    
    return rnn,training_history 
    

#%% main function
if __name__ == '__main__':
    # import Eksplorativna_analiza
    # data = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    rnn,training_history = GetSimpleRNNWithTrainingHistory(epochs=3)
    
    
    
    
#%% old
    
    # #%%
    # # from keras.src import ops
    # # from keras.src.layers import RNN
    # import tensorflow as tf
    # import keras
    
    # #%%
    # # First, let's define a RNN Cell, as a layer subclass.
    # class MinimalRNNCell(keras.layers.Layer):
    
    #     def __init__(self, units, **kwargs):
    #         super().__init__(**kwargs)
    #         self.units = units
    #         self.state_size = units
    
    #     def build(self, input_shape):
    #         self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
    #                                       initializer='uniform',
    #                                       name='kernel')
    #         self.recurrent_kernel = self.add_weight(
    #             shape=(self.units, self.units),
    #             initializer='uniform',
    #             name='recurrent_kernel')
    #         self.built = True
    
    #     def call(self, inputs, states):
    #         prev_output = states[0]
    #         h = tf.matmul(inputs, self.kernel)
    #         output = h + tf.matmul(prev_output, self.recurrent_kernel)
    #         return output, [output]
    
    # # Let's use this cell in a RNN layer:
    
    # #%% simple rnn 1 cell
    # # cell = MinimalRNNCell(32)
    # # x = keras.Input((None, 5))
    # # layer = keras.layers.RNN(cell)
    # # y = layer(x)
    
    # #%% my rnn
    # cell = MinimalRNNCell(32)
    # cells = [MinimalRNNCell(32), MinimalRNNCell(64), MinimalRNNCell(64)]
    # # mymodel = keras.models.Sequential()
    # # mymodel.add(keras.layers.RNN(cell))
    # # mymodel.build((1, 10, 1))
    # # print(mymodel.summary())
    # inputlayer = keras.Input((10, 1))
    # rnnlayer = keras.layers.RNN(cells) (inputlayer)
    # denselayer = keras.layers.Dense(4) (rnnlayer)
    # outputlayer =  denselayer
    
    # mymodel = keras.Model(inputlayer , outputlayer )
    # print(mymodel.summary())
    
    # #%% fitting
    
    
    # #%% stacked
    # # Here's how to use the cell to build a stacked RNN:
    
    # cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    # x = keras.Input((None, 5))
    # layer = keras.layers.RNN(cells)
    # y = layer(x)
