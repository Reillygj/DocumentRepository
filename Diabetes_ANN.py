
import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import TimeDistributed
import tensorflow as tf
from tensorflow import keras
#Activating Tuner randomsearch
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

import keras_tuner as kt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
"""            USE GPU TO RUN MODEL  """
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    

###########################################################################
"""                           READ IN DATA     
Clean Data - Remove Nulls, Address/Impute Outliers, Scale Data if needed, 
Make sure dataset is balanced
"""
#Data is cleaned and balanced already, above steps not needed
dataframe = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv', delimiter = ",")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataframe.head(2)
dataframe.describe()
dataframe.info()
dataframe.shape

# create X & Y columns
x = dataframe.drop('Diabetes_binary', axis = 'columns').values

#Values because it's going to be converted to categorical later in code 
y = dataframe['Diabetes_binary'].values

#############################################################################
"""             ADDRESS CATEGORICAL VALUES                          """

y = tf.keras.utils.to_categorical(y)

###########################################################################
"""                      TRAIN/TEST SPLIT                               """

#Training / Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=123)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Use Keras Tuner to optimize hyper-parameters.  Units range from 5 -100, 
with a step of 5 units.  Activation functions are relu, tanh, and swish.  
Optimizaiton functions are Adam, RMSProp, and SGD"""


##############################################################################
"""                             LEARNING RATE                         """

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)


##############################################################################
"""                      MODEL AND OPTIMIZATION PARAMETERS                  """

def build_model(hp):
    model=Sequential()
    
    model.add(Dense(units=hp.Int("units", min_value=10, max_value=100, step=5),
                    input_dim=21 ,
                   activation=hp.Choice("activation", ["relu", "tanh", "swish"]))) 
    
    model.add(Dense( units=hp.Int("units1", min_value=10, max_value=200, step=5), 
                    activation=hp.Choice("activation", ["relu", "tanh", "swish"])))
        
    model.add(Dense( units=hp.Int("units2", min_value=10, max_value=200, step=5),
                     activation=hp.Choice("activation", ["relu", "tanh", "swish"])))
      
    model.add(Dense( units=hp.Int("units3", min_value=10, max_value=200, step=5),
                        activation=hp.Choice("activation", ["relu", "tanh", "swish"])))
   
        
    model.add(Dense( units=hp.Int("units4", min_value=10, max_value=200, step=5),
                      activation=hp.Choice("activation", ["relu", "tanh", "swish"])))
   
    # two output variables
    model.add(Dense(2, activation=hp.Choice("activation",['sigmoid','hard_sigmoid'])))  #  Binary output Heart Disease Yes/No requires sigmoid
       
    model.compile(loss='binary_crossentropy', 
                  optimizer=hp.Choice("optimizer", ["adam", "RMSProp", "SGD"]),
                  metrics=["accuracy"])
    return model


###############################################################################

"""                 TUNER OPTIMIZER USING HYPERBAND       """

#Directory to save Model outputs
my_dir = 'C:/Users/gabri/Desktop/AI_&_Neural_Networks/M6/PROJECT DATA/'

tuner = kt.Hyperband(
    build_model,
    objective= 'val_accuracy',
    directory='my_dir',
    overwrite=True,
    max_epochs=10,
    )
    

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

#Training Tuner for Hyperband
tuner.search(x_train, y_train, epochs=30, validation_split = 0.2)

# Get best hyperparameters for Hyperband
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
hypermodel = tuner.hypermodel.build(best_hps)

# Run model to establish loss and accuracy again
start_time = datetime.datetime.now()
history = hypermodel.fit(x_train,y_train, epochs=30, validation_split=0.2)
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

#Evaluate Model
hypermodel.evaluate(x_test,y_test)

# Overview of training
tuner.results_summary()
hypermodel.summary()
# Trial name based on tuner model
best_model_in_directory = tuner.oracle.get_trial('0df0d5e5c4637cb42e0a7dedb232ffac')
best_model_in_directory.summary()

########################################################################
"""                   CONFUSION MATRIX  HYPERBAND        """

##Validate data 
from sklearn.metrics import confusion_matrix
#labels = list(y_test)
y_test_arg=np.argmax(y_test,axis=1)
predictions = np.argmax(hypermodel.predict(x_test),axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_arg, predictions))
cm=confusion_matrix(y_test_arg, predictions)

import matplotlib.pyplot as pyplot

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Hyperband Confusion matrix',
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pyplot.figure()
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

# 1 = has diabetes /  0 = no diabetes
plot_confusion_matrix(cm, classes = [0,1]) # dataset is binary and ouput variables are 0,1


########################################################################
"""                   ACCURACY / LOSS GRAPH   HYPERBAND               """ 

loss = history.history # data points of what was plotted
loss
abc = min(loss['val_loss'])
xyz = abc
plt.plot(loss['loss'], label = 'Training_Loss', marker='o', color='red')
plt.plot(loss['val_loss'], label = 'Loss', marker='o', color='blue')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Hyperband Training vs. Prediction Loss')
plt.legend()
plt.text(5.55, .55,'Loss Minimum: {:.2f}'.format(xyz))
plt.show () 


loss = history.history # data points of what was plotted
loss
abc = max(loss['val_accuracy'])
xyz = abc*100
plt.plot(loss['val_accuracy'], label = 'Val_Accuracy', marker='o', color='blue')
plt.plot(loss['accuracy'], label = 'Training_Accuracy', marker='o', color='red')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Hyperband Training vs. Prediction Accuracy')
plt.legend()
plt.text(6.55, .74,'Max Prediction Accuracy: {:.2f}'.format(xyz))
plt.show () 




print ("Time required for training:",stop_time - start_time)

###############################################################################
###############################################################################
"""                 TUNER OPTIMIZER USING RANDOMSEARCH         """

my_dir1 = 'C:/Users/gabri/Desktop/AI_&_Neural_Networks/M6/PROJECT DATA/'

tuner1 = kt.RandomSearch(
    build_model,
    objective= 'val_accuracy',
   # direction='min',
    directory='my_dir1',
    overwrite=True,
    max_trials=5)

#Training Tuner for Randomsearch
tuner1.search(x_train, y_train, epochs=30, validation_split = 0.2)

#Getting Best model [indexed first, compare to tuner.summary and it's the same]
models = tuner1.get_best_models(num_models=3)
best_model = models[0]  #Models are indexed by best objective return to worse


#Train best model using entire dataset 

start_time1 = datetime.datetime.now()
history1 = best_model.fit(x_train,y_train, epochs=30, validation_split=0.2)
stop_time1 = datetime.datetime.now()
print ("Time required for training:",stop_time1 - start_time1)

#Evaluate Model
best_model.evaluate(x_test,y_test)

# Overview of training
best_model.summary()
tuner1.results_summary()

########################################################################
"""                   CONFUSION MATRIX RANDOM SEARCH       """

##Validate data 
from sklearn.metrics import confusion_matrix
#labels = list(y_test)
y_test_arg=np.argmax(y_test,axis=1)
predictions = np.argmax(best_model.predict(x_test),axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_arg, predictions))
cm=confusion_matrix(y_test_arg, predictions)

import matplotlib.pyplot as pyplot

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Randomsearch Confusion matrix',
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pyplot.figure()
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

plot_confusion_matrix(cm, classes = [0,1]) # dataset is binary and ouput variables are 0,1

########################################################################
"""                   ACCURACY / LOSS GRAPH  RANDOM SEARCH              """ 

loss1 = history1.history # data points of what was plotted
loss1
abc1 = min(loss['val_loss'])
xyz1 = abc1
plt.plot(loss1['loss'], label = 'Training_Loss',marker='o', color='red')
plt.plot(loss1['val_loss'], label = 'Loss',marker='o', color='green')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Randomsearch Training vs. Prediction Loss')
plt.legend()
plt.text(5.55, .53,'Loss Minimum: {:.2f}'.format(xyz1))
plt.show () 


loss2 = history1.history # data points of what was plotted
loss2
abc2 = max(loss2['val_accuracy'])
xyz2 = abc*100
plt.plot(loss2['val_accuracy'], label = 'Val_Accuracy',marker='o', color='green')
plt.plot(loss2['accuracy'], label = 'Training_Accuracy',marker='o', color='red')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Randomsearch Training vs. Prediction Accuracy')
plt.legend()
plt.text(6.55, .75,'Max Prediction Accuracy: {:.2f}'.format(xyz2))
plt.show () 



stop_time = datetime.now()
print ("Time required for training:",stop_time - start_time)

##############################################################################
###############################################################################
"""         REPRODUCE MODEL & VERIFY HYPER-PARAMETERS   (HARD_SIGMOID)       """

###############################################################################
"""   Optimizer  for chosen model to refine loss/accuracy"""
opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

##############################################################################

model2=Sequential()

model2.add(Dense(units=100,
                input_dim=21 ,
               activation= "swish")) 

model2.add(Dense(units=70, activation = "swish"))  

model2.add(Dense(units=20, activation = "swish"))   
model2.add(Dense(units=105, activation = "swish"))     
# model.add(Dropout(.25))
    
model2.add(Dense(units=50, activation = "swish"))     

model2.add(Dense(2, activation='hard_sigmoid'))  #  Binary output Heart Disease Yes/No requires sigmoid
   
model2.compile(loss='binary_crossentropy', 
              optimizer='Adam',
              metrics=["accuracy"])

test = model2.fit(x_train, y_train, epochs=30, validation_split = 0.2)

"""        Graph Loss/Accuracy/ Confusion Matrix                     """
from sklearn.metrics import confusion_matrix
#labels = list(y_test)
y_test_arg=np.argmax(y_test,axis=1)
predictions = np.argmax(model2.predict(x_test),axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_arg, predictions))
cm=confusion_matrix(y_test_arg, predictions)

#############
loss3 = test.history # data points of what was plotted
loss3
abc = min(loss3['val_loss'])
xyz = abc
plt.plot(loss3['loss'], label = 'Training_Loss', marker='o', color='red')
plt.plot(loss3['val_loss'], label = 'Loss', marker='o', color='blue')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Hard Sigmoid Training vs. Prediction Loss')
plt.legend()
plt.text(5.55, .55,'Loss Minimum: {:.2f}'.format(xyz))
plt.show () 


loss3 = test.history # data points of what was plotted
loss3
abc = max(loss3['val_accuracy'])
xyz = abc*100
plt.plot(loss3['val_accuracy'], label = 'Val_Accuracy', marker='o', color='blue')
plt.plot(loss3['accuracy'], label = 'Training_Accuracy', marker='o', color='red')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Hard Sigmoid Training vs. Prediction Accuracy')
plt.legend()
plt.text(6.55, .74,'Max Prediction Accuracy: {:.2f}'.format(xyz))
plt.show () 

###############################################################################
"""         REPRODUCE MODEL & VERIFY HYPER-PARAMETERS   (SIGMOID)       """
model2=Sequential()

model2.add(Dense(units=100,
                input_dim=21 ,
               activation= "swish")) 

model2.add(Dense(units=70, activation = "swish"))  

model2.add(Dense(units=20, activation = "swish"))   
model2.add(Dense(units=105, activation = "swish"))     
# model.add(Dropout(.25))
    
model2.add(Dense(units=50, activation = "swish"))     

model2.add(Dense(2, activation='sigmoid'))  #  Binary output Heart Disease Yes/No requires sigmoid
   
model2.compile(loss='binary_crossentropy', 
              optimizer="adam",
              metrics=["accuracy"])
test = model2.fit(x_train, y_train, epochs=30, validation_split = 0.2)

"""        Graph Loss/Accuracy/ Confusion Matrix                     """
from sklearn.metrics import confusion_matrix
#labels = list(y_test)
y_test_arg=np.argmax(y_test,axis=1)
predictions = np.argmax(model2.predict(x_test),axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_arg, predictions))
cm=confusion_matrix(y_test_arg, predictions)

#############
loss3 = test.history # data points of what was plotted
loss3
abc = min(loss3['val_loss'])
xyz = abc
plt.plot(loss3['loss'], label = 'Training_Loss', marker='o', color='red')
plt.plot(loss3['val_loss'], label = 'Loss', marker='o', color='blue')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Sigmoid Training vs. Prediction Loss')
plt.legend()
plt.text(5.55, .52,'Loss Minimum: {:.2f}'.format(xyz))
plt.show () 


loss3 = test.history # data points of what was plotted
loss3
abc = max(loss3['val_accuracy'])
xyz = abc*100
plt.plot(loss3['val_accuracy'], label = 'Val_Accuracy', marker='o', color='blue')
plt.plot(loss3['accuracy'], label = 'Training_Accuracy', marker='o', color='red')
#plt.axvline(xyz, color='k', linestyle='dashed', linewidth=1)
plt.title('Sigmoid Training vs. Prediction Accuracy')
plt.legend()
plt.text(6.55, .74,'Max Prediction Accuracy: {:.2f}'.format(xyz))
plt.show () 

