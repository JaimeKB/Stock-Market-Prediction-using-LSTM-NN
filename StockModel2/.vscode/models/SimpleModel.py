import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import gc
import sys

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """
    history_size is the size of the past window of information
    target_size is how far in the future the model will predict, in this case 1
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
        
    return np.array(data), np.array(labels)

#############################


def create_time_steps(length):
    return list(range(-length, 0))

##########################

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                   label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
    
    return plt


###########################




def baseline(history):
    return np.mean(history)

############################







if __name__ == "__main__":
    mpl.rcParams['figure.figsize'] = (17, 5)
    mpl.rcParams['axes.grid'] = False
    sns.set_style("whitegrid")

    ############################

    # Data Loader Parameters
    BATCH_SIZE = 32
    BUFFER_SIZE = 100
    TRAIN_SPLIT = 1000

    # LSTM Parameters
    EVALUATION_INTERVAL = 200
    EPOCHS = 4
    PATIENCE = 5

    # Reproducibility
    SEED = 13
    tf.random.set_seed(SEED)

    ############################

    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    print("DataFrame Shape: {} rows, {} columns".format(*df.shape))
    print(df.head())

    # #############################

    # #############################

    uni_data = df['Close']
    uni_data.index = df['Date']
    print()
    print(uni_data.head())

    # #############################

    # uni_data.plot(subplots=True)
    plt.show()
    uni_data = uni_data.values

    # ##############################

    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()

    # #############################

    uni_data = (uni_data-uni_train_mean)/uni_train_std

    # ############################

    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(dataset=uni_data,
                                            start_index=0,
                                            end_index=TRAIN_SPLIT,
                                            history_size=univariate_past_history,
                                            target_size=univariate_future_target)
  
    x_val_uni, y_val_uni = univariate_data(dataset=uni_data,
                                        start_index=TRAIN_SPLIT,
                                        end_index=None,
                                        history_size=univariate_past_history,
                                        target_size=univariate_future_target)

    # ###########################

    print("In:")
    print(uni_data.shape)
    print(uni_data[:5])

    print("\nOut")
    print(x_train_uni.shape)


    print(x_train_uni.shape[0] / uni_data.shape[0])

    # ############################

    print ('Single window of past history. Shape: {}'.format(x_train_uni[0].shape))
    print (x_train_uni[0])
    print ('\n Target temperature to predict. Shape: {}'.format(y_train_uni[0].shape))
    print (y_train_uni[0])



    # ############################

    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    plt.show()

    # #############################

    # Sending in the mean average of the past 20 to act as a baseline to beat
    show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')
    # plt.show()

    # ############################

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


    print("THE STUFF I WANT")
    print(train_univariate)
    print("AND")
    print(val_univariate)


    # #############################

    print(x_train_uni.shape)


    # ##############################

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    # ###########################

    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)

    # #############################

    early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
    simple_lstm_model.fit(train_univariate,
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=val_univariate,
                        callbacks=[early_stopping],
                        validation_steps=50)

    # ##############################

    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                        simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()

    # ############################