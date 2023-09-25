
import pickle
import datetime as dt
import keras
from keras import Model
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Reshape, Dropout, LSTM, Input, \
    TimeDistributed, Conv2D, MaxPooling2D, ConvLSTM2D
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
# Load the data: Whole dataset
with open('/data/amatin/Pickle_data/data_40_Size_112x112_Train_0.8_Test_0.2.pkl', 'rb') as f:
    (features_train, features_test, labels_train, labels_test) = pickle.load(f)

# Load the data partial for training and Test
#with open('/data/amatin/Pickle_data/data.pkl', 'rb') as f:
    #(features_train, features_test, labels_train, labels_test) = pickle.load(f)

no_classes=12
D = 40   #Number of frames.
W = 112  #Frame Width.
H = 112  #Frame Height.
C = 3    #Number of channels.
sample_shape = (D, W, H, C) #Single Video shape.

batch_size1 = 32
no_epochs = 50
learning_rate = 0.0001
validation_split = 0.2
verbosity = 1

#######################    ***    Visualization  ***#########################
def plot_history(history, name):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(5.5, 9.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
  plt.savefig('/home/amatin/Data/gpu_test/'+name+'_performance.eps')
  #plt.show()
###########################################################################################

'''convlstm_3D_CNN_old'''
# Create a convLSTM to process the features extracted by the CNN

convlstm_3D_CNN_old = keras.models.Sequential([

    keras.layers.ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape=(D, W, H, C)),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.TimeDistributed(Dropout(0.2)),

    keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True),
    keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.TimeDistributed(Dropout(0.2)),
    keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((2, 2, 2)),
    keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    keras.layers.MaxPooling3D((2, 2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(12, activation='softmax')
])

# ** Run Hybrid convlstm_3D_CNN
convlstm_3D_CNN_old.summary()

# Optional - For Debugging
plot_model(convlstm_3D_CNN_old, to_file = 'convlstm_3D_CNN_old_structure_plot.png', show_shapes = True, show_layer_names = True)

#early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

convlstm_3D_CNN_old.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

convlstm_3D_CNN_old_model_training_history = convlstm_3D_CNN_old.fit(x=features_train, y=labels_train, epochs=no_epochs, batch_size= batch_size1, shuffle=True,
                                             validation_split = validation_split)
# Evaluate the trained model.
convlstm_3D_CNN_old_evaluation_history = convlstm_3D_CNN_old.evaluate(features_test, labels_test)

# Get the loss and accuracy from model_evaluation_history.
convlstm_3D_CNN_old_evaluation_loss, convlstm_3D_CNN_old_evaluation_accuracy = convlstm_3D_CNN_old_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
convlstm_3D_CNN_old_model_file_name = f'convlstm_3D_CNN_old_model___Date_Time_{current_date_time_string}___Loss_{convlstm_3D_CNN_old_evaluation_loss}___Accuracy_{convlstm_3D_CNN_old_evaluation_accuracy}.h5'

# Save the Model.
convlstm_3D_CNN_old.save(convlstm_3D_CNN_old_model_file_name)

name='convlstm_3D_CNN_old_model_training_history_10'
plot_history(convlstm_3D_CNN_old_model_training_history,name)

print('################################## ** Finished convlstm_3D_CNN_old Model  ** ###################################################')
######################################################################################
######################################################################################


################################################################################################
################################################################################################
def create_LRCN_model():
    
    # We will use a Sequential model for model construction.
    model = Sequential()
    # Define the Model Architecture.
    ########################################################################################################################

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),input_shape=(D, W, H, C)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(no_classes, activation='softmax'))
    # Display the models summary.
    model.summary()
    return model

"""**Run_LRCN**"""
# Construct the required LRCN model.
LRCN_model = create_LRCN_model()

# Display the success message.
print("LRCN_Model Created Successfully!")

# Plot the structure of the contructed LRCN model.
plot_model(LRCN_model, to_file='LRCN_model_structure_plot.png', show_shapes=True, show_layer_names=True)

# Create an Instance of Early Stopping Callback.
#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)

# Compile the model and specify loss function, optimizer and metrics to the model.
LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# Start training the model.
LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=no_epochs, batch_size= batch_size1, shuffle=True,
                                             validation_split = validation_split)

# Evaluate the trained model.
LRCN_model_evaluation_history = LRCN_model.evaluate(features_test, labels_test)

# Get the loss and accuracy from model_evaluation_history.
LRCN_model_evaluation_loss, LRCN_model_evaluation_accuracy = LRCN_model_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
LRCN_model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{LRCN_model_evaluation_loss}___Accuracy_{LRCN_model_evaluation_accuracy}.h5'

# Save the Model.
LRCN_model.save(LRCN_model_file_name)

name='LRCN_model_training_history_10'
plot_history(LRCN_model_training_history,name)

print('################################## ** Finished LRCN Model ** ###################################################')
###########################################################################################

#early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

convlstm_3D_CNN_simplified.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

convlstm_3D_CNN_simplified_model_training_history = convlstm_3D_CNN_simplified.fit(x=features_train, y=labels_train, epochs=no_epochs, batch_size= batch_size1, shuffle=True,
                                             validation_split = validation_split)
# Evaluate the trained model.
convlstm_3D_CNN_simplified_evaluation_history = convlstm_3D_CNN_simplified.evaluate(features_test, labels_test)

# Get the loss and accuracy from model_evaluation_history.
convlstm_3D_CNN_simplified_evaluation_loss, convlstm_3D_CNN_simplified_evaluation_accuracy = convlstm_3D_CNN_simplified_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
convlstm_3D_CNN_simplified_model_file_name = f'convlstm_3D_CNN_simplified_model___Date_Time_{current_date_time_string}___Loss_{convlstm_3D_CNN_simplified_evaluation_loss}___Accuracy_{convlstm_3D_CNN_simplified_evaluation_accuracy}.h5'

# Save the Model.
convlstm_3D_CNN_simplified.save(convlstm_3D_CNN_simplified_model_file_name)

name='convlstm_3D_CNN_simplified_model_training_history_10'
plot_history(convlstm_3D_CNN_simplified_model_training_history,name)

print('################################## ** Finished convlstm_3D_CNN_simplified Model  ** ###################################################')
