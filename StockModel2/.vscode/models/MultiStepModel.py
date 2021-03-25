import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import gc
import sys

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size # saying start at 0 + 60 = 60
    if end_index is None:
        end_index = len(dataset) - target_size

    end_index = end_index - target_size

    for i in range(start_index, end_index):    # in range of 60 to 1000
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

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
    # SEED = 13
    # tf.random.set_seed(SEED)

    ############################

    df=pd.read_csv("C:/Users/Jaime Kershaw Brown/Documents/Final year project/TSLA.csv")

    print("DataFrame Shape: {} rows, {} columns".format(*df.shape))
    print(df.head())

    # #############################

    # #############################

    features_considered = ['Close', 'Open', 'High', 'Low']
    features = df[features_considered]
    features.index = df['Date']

    print()
    print(features.head())

    # # #############################

    # features.plot(subplots=True)
    # plt.show()
    dataset = features.values

    # # ##############################

    sc = MinMaxScaler(feature_range = (0, 1))
    dataset = sc.fit_transform(dataset[:TRAIN_SPLIT])

    # # ############################

    past_history = 60
    future_target = 0
    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target, STEP,
                                                    single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                TRAIN_SPLIT, None, past_history,
                                                future_target, STEP,
                                                single_step=True)
    # # ###########################

    print(x_train_single.shape)
    print ('Single window of past history : {}'.format(x_train_single[0].shape))
    print(x_train_single.shape[-2:])

    # # ############################

    # train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    # train_data_single = train_data_single.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    # val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    # val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    # print(val_data_single)

    ################################

    # single_step_model = tf.keras.models.Sequential()
    # single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
    
    # single_step_model.add(tf.keras.layers.Dense(1))

    # single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# ##############################

    # for x, y in val_data_single.take(1):
    #     print("HERE")
    #     print(single_step_model.predict(x).shape)

# ###########################

#     print(f"Evaluation Threshold: {EVALUATION_INTERVAL}",
#         f"Epochs: {EPOCHS}", sep="\n")

#     early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
#     single_step_history = single_step_model.fit(train_data_single,
#                                                 epochs=EPOCHS,
#                                                 steps_per_epoch=EVALUATION_INTERVAL,
#                                                 validation_data=val_data_single,
#                                                 callbacks=[early_stopping],
#                                                 validation_steps=50)


###########################




###########################



###########################