#%% imports
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from RNN_data_preprocessing import *


#%% class declaration

class MultiInputRNN:
    def __init__(self, RNN_input_shape, Dense_input_shape, output_len, loss='mean_squared_error'):
        self.RNN_input_shape = RNN_input_shape
        self.Dense_input_shape = Dense_input_shape
        self.output_len = output_len
        self.loss = loss
        self.model = None
        self.RNNInputLayer = None
        self.RNNLayer = None
        self.DenseInputLayer = None
        self.DenseLayer = None
        self.ConcatLayer = None
        self.DenseOutputLayer = None
        self._addLayers()
        self._compileModel()
        
    def _addLayers(self):
        self.RNNInputLayer = keras.layers.Input(shape=self.RNN_input_shape)
        self.RNNLayer = keras.layers.GRU(units=10) (self.RNNInputLayer )
        
        self.DenseInputLayer = keras.layers.Input(shape=self.Dense_input_shape)
        self.DenseLayer = keras.layers.Dense(8,activation='linear')(self.DenseInputLayer )
        
        ## trainable=True by default
        self.ConcatLayer = keras.layers.Concatenate(trainable=True)([self.DenseLayer ,self.RNNLayer ])
        self.DenseOutputLayer = keras.layers.Dense(self.output_len,activation='linear')(self.ConcatLayer )
        
        self.model = keras.models.Model(
            inputs=[self.RNNInputLayer, self.DenseInputLayer]
            ,outputs=self.DenseOutputLayer 
            )
    
    def _compileModel(self):
        self.model.compile(loss=self.loss
                           , optimizer="adam"
                           , metrics=[keras.metrics.MeanSquaredError()])
    
        

#%% function definition
def GetSimpleMaskedRNNWithTrainingHistory(df = None,epochs=3, timeseries_batch_size= 10, timepoints_per_day= 24 ):
    ''' returns the  'SimpleRNN' object with its training history 
    'timeseries_sequence_length' can be set to practicaly anything since we use the time difference in minutes between instances'''

    
    if df == None:
        # df = pd.read_csv('masked_RNN_input_value0_masked_rows_24_timepoints.csv')
        # df['DATE TIME OCC'] = pd.to_datetime(df['DATE TIME OCC']
        #                                        ,format='%Y-%m-%d %H:%M:%S'
        #                                       ,errors='coerce'
        #                                       )
        df = GetDataFrameWithMask()
        # print(df)
        ## NOTE: KEEP THE ''DATE TIME OCC' in its original state since we will pass it to the RNN without skipping it with a mask
        # df.loc[df['AVAILABLE MASK'] == 0] = 0 ## set the mask values to be 0
        # print(df)
    

    ## %% train and validation split
    train_to_validation_split = 0.9
    train_df = df
    
    validation_df = train_df[ int(len(train_df) * train_to_validation_split) : ]
    train_df = train_df[ 0 : int(len(train_df) * train_to_validation_split) ]
    
    
    ##%% create time series
    ## these 2 paramaters are now function paramaters
    timepoints_per_day = 24 ## from above so we look at the previous day
    timeseries_sequence_length = timepoints_per_day 
    timeseries_batch_size = timepoints_per_day
    
    #####################
    ## RNN INPUT (can't put timeseries object, they must be tensors)
    rnn_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    rnn_target_columns = ['LAT','LON']
    
    ## X value for RNN training
    rnn_train_data_input = np.array(train_df[rnn_input_columns ]).copy()
    
    ## the dimensions of the input
    dim1_train_data =  int(rnn_train_data_input.size  / (timepoints_per_day*len(rnn_input_columns))) ##how many instances we have
    dim2 = timepoints_per_day   ##sequence length
    dim3 = len(rnn_input_columns) ##number of fields/columns
    
    ## drop excess data so we have only fully-filled timeserieses 
    rnn_train_data_input = rnn_train_data_input[ 0: dim1_train_data * dim2 ]  
    rnn_train_data_input = rnn_train_data_input.reshape( 
        ( dim1_train_data
         ,dim2
         ,dim3 
         ))
    
    ## Y value for RNN training
    rnn_train_data_output = np.array(train_df[rnn_target_columns ]).copy()
    rnn_train_data_output = rnn_train_data_output[dim2::dim2]
    
    
    ##Get the RNN validation input and output data
    ## X value for RNN validation
    rnn_validation_data_input = np.array(validation_df[rnn_input_columns ]).copy()
    
    ## how many timeserieses we have 
    dim1_validation_data =  int(rnn_validation_data_input.size  / (timepoints_per_day*len(rnn_input_columns)))
    
    ## drop excess data so we have only fully-filled timeserieses 
    rnn_validation_data_input = rnn_validation_data_input[ 0: dim1_validation_data * dim2 ]
    rnn_validation_data_input = rnn_validation_data_input.reshape( 
        ( dim1_validation_data 
         ,dim2
         ,dim3 
         ))
    
    ## Y value for RNN validation
    rnn_validation_data_output = np.array(validation_df[rnn_target_columns ]).copy()
    rnn_validation_data_output = rnn_validation_data_output[dim2::dim2]
    
    
    #####################
    ## DENSE LAYER INPUT
    dense_input_columns = ['YEAR OCC','MONTH OCC','DAY OCC','HOUR OCC','AREA','Crm Cd']
    dense_target_columns = rnn_target_columns ## must be the same as target columns
    
    ## X value for Dense NN training
    dense_train_data_input = np.array(train_df[dense_input_columns ][dim2::dim2] ) ## WILL CRASH IF YOU DON'T CHECK THE DF SHAPES
    ## X value for Dense NN validation
    dense_validation_data_input = np.array(validation_df[dense_input_columns][dim2::dim2])
    ##NOTE: 
    ## Y value for Dense NN validation and training is 'rnn_train_data_input' and 'rnn_validation_data_output'
    
    
    ##########################
    ##%% create and compile the model
    multi_rnn = MultiInputRNN(RNN_input_shape= ( timeseries_sequence_length,len(rnn_input_columns ))
                              , Dense_input_shape= (len(dense_input_columns ))
                              , output_len=len(dense_target_columns )  
                          )
    print(multi_rnn.model.summary())
    
    
    ##########################
    ##%% train
    training_history = multi_rnn.model.fit( x=[rnn_train_data_input ,dense_train_data_input ]
                                            , y= rnn_train_data_output 
                                            , epochs= epochs
                                            , shuffle= False
                                            , validation_data= ( [ rnn_validation_data_input,dense_validation_data_input] , rnn_validation_data_output) 
                                            # , batch_size=16
                                            )
        
    return multi_rnn ,training_history 
        
    
#%% main function
if __name__ == '__main__':
    # import Eksplorativna_analiza
    # data = Eksplorativna_analiza.izvrsi_eksplorativnu_analizu()
    multi_rnn, training_hitstory = GetSimpleMaskedRNNWithTrainingHistory(epochs=100)
