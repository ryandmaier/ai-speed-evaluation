from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import backend as K
import time
import datetime
from datetime import timezone
import csv
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

num_trains = 100
log_root = 'Local_ml_test_logs_'

def train_model(log_writer):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    batch_size = 128
    num_classes = 10
    epochs = 30

    for i in range(num_trains):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        t1 = time.time()
        hist = model.fit(
            x_train, 
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test)
        )
        t2 = time.time()
        print('Time elapsed for fit:',str(t2-t1))
        score = model.evaluate(x_test, y_test, verbose=0)

        # log_writer.writerow(['Datetime','StartTime','EndTime','Seconds','Minutes','TrainSize','TestSize','Epochs','BatchSize','Loss','Accuracy','Precision','Recall'])
        row = [dt_now(),t1,t2,t2-t1,(t2-t1)/60,len(x_train),len(x_test),epochs,batch_size, score[0],score[1],score[2],score[3]]
        log_writer.writerow(row)
        print('\nModel trained and logged!')
        print(row)
        print('\n\n\n\n')

def dt_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))).strftime("%Y%m%d_%H%M%S")

def start_test():
    print("Begining Test Execution...")
    t_start = dt_now()
    print('Start time: '+str(t_start))
    log_file = log_root+str(t_start)+'.csv'
    print('Creating log file: '+log_file)
    with open(log_file, mode='w') as csv_file:
        log_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(['Datetime','StartTime','EndTime','Seconds','Minutes','TrainSize','TestSize','Epochs','BatchSize','Loss','Accuracy','Precision','Recall'])
        train_model(log_writer)

if __name__ == "__main__":
    start_test()