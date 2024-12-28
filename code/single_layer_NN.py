#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *

#%% 

class SimpleSingleLayerNN:
    def __init__(self, input_shape, output_len, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = keras.models.Sequential()
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.model.add(keras.layers.Dense(16,
                                          input_dim= self.input_shape,
                                          activation=keras.activations.relu
                                          ))
        self.model.add(keras.layers.Dense(self.output_len))
    
    def _compileModel(self):
        self.model.compile(loss=self.loss, optimizer="adam")

#%%
def GetSimple1LayerNNWithTrainingHistory(df=None,epochs=3):

    ##%% data preprocessing
    if (df == None):
        import Eksplorativna_analiza
        df = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
        
    df = GetExactTimeOfCrimeOccurrenceInMinutesSinceEpoch(df,new_column_name = 'TOTAL TIME OCC minutes')
    df = GetExactTimeOfCrimeOccurrenceInMinutes(df,new_column_name = 'TIME OCC minutes')
    
    ##%% train and validation split
    train_to_validation_split = 0.9
    train_df = df.loc[ (df['DATE OCC'].dt.year >= 2023) 
                      & (df['DATE OCC'].dt.year <= 2024) ]
    # train_df  = df
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    ##%% 
    input_columns = ['YEAR OCC','MONTH OCC','DAY OCC', 'HOUR OCC','AREA','Crm Cd']
    target_columns = ['LAT','LON']
    
    
    train_data_inputs = train_df[input_columns]
    train_data_targets = train_df[target_columns ]
    
    validation_data_inputs = validation_df[input_columns]
    validation_data_targets = validation_df[target_columns ]
    
    ##%% create and compile
    nn = SimpleSingleLayerNN(input_shape= len(input_columns)  
                             ,output_len=len(target_columns )  
                             )
    print(nn.model.summary())
    
    ##%% train
    training_history = nn.model.fit(train_data_inputs 
                                    , train_data_targets 
                                    , epochs= epochs
                                    , shuffle= True
                                    , validation_data= (validation_data_inputs , validation_data_targets)
                                    )
    
    return nn,training_history 
   
#%% main function
if __name__ == '__main__':
    # import Eksplorativna_analiza
    # data = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    nn,training_history = GetSimple1LayerNNWithTrainingHistory(epochs=3)
    
