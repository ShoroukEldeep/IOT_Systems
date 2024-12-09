
# importing libs
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import os, json
from customCallback import customCallback
from utils.plotting import plot_confusion_matrix, plot_history
from tqdm import tqdm
from customLoss import custom_binary_crossentropy,compute_overall_accuracy
# from hyperdash import Experiment
# from utils.gpu import get_gpu_memory
# from utils.cpu import get_cpu_memory
#import mycallback
# CLASSES = [i for i in range(0,7)]
CLASSES = ['backdoor', 'ddos', 'injection', 'password', 'ransomware', 'xss']

tf.get_logger().setLevel('WARNING')

# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# def get_batch_size(para_count, gpu=False):
#     if gpu:
#         bs = get_gpu_memory() / 4
#     else:
#         bs = get_cpu_memory() / 4
#     n = para_count
#     return int(bs / n)

def buildModel(para):
    # with tpu_strategy.scope():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((10,2),input_shape=(10,2)))
    for i in range(0,int(para[0])):
        model.add(tf.keras.layers.LSTM(units=int(para[1+(2*i)]),recurrent_dropout=(int(para[2+(2*i)])/100),return_sequences=True))
    do = para[9] / 100
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(do))
    for i in range(0,int(para[10])):
        model.add(tf.keras.layers.Dense(int(para[11+i]),activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(len(CLASSES),activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model,0

def makeCallbacks(filename,monitor='loss',patience=2):
    CP = tf.keras.callbacks.ModelCheckpoint(filename, monitor=monitor, verbose=1,
    save_best_only=True, mode='auto',save_freq="epoch")
    LR = tf.keras.callbacks.ReduceLROnPlateau(patience=patience, verbose=1)
    CC = customCallback(filename)
    return [CP,LR,CC]

def enterFile(filename, goBack=False):
    os.chdir(filename)
def fileCreation():
    filename = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.mkdir(filename)
    enterFile(filename)
    weights_path = "weights"+str(datetime.now().strftime("%H_%M_%S"))+".hdf5"
    history_path = "history"+str(datetime.now().strftime("%H_%M_%S"))+".json"
    return filename,history_path,weights_path
def plotCM(model,x, y,label):
    # y_pred = np.argmax(model.predict(x), axis=-1)
    y_pred = model.predict(x)
    y = np.array(y)
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    # y = np.argmax(y, axis=-1)multilabel_confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    plot_confusion_matrix(cm=cm, classes=CLASSES, title='Confusion Matrix',label=str(label))
def apply_confusion_matrix(model,data):
    for i in range(0,len(data)):
        # pl(model, data[i][0], data[i][1])
        plotCM(model,data[i][0], data[i][1],i)
def set(X, Y, home_file, para):
    os.chdir(home_file)
    filename,history_path, weights_path = fileCreation()
    x_train,x_val,y_train,y_val = train_test_split(X,Y,shuffle=True,random_state=42,train_size=0.9)
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,shuffle=True,random_state=42,train_size=0.9)
    # x_test = x_val
    # y_test = y_val
    #Building the model
    print(np.argmax(y_test,axis=1))
    model,para_count = buildModel(para)
    model.summary()
    #training
    # bs = 16 * tpu_strategy.num_replicas_in_sync
    bs = 32
    print("batch size = " ,bs)
    try:
        # exp = Experiment("iot")
        history = model.fit(x_train, y_train,
                            epochs=150,
                            batch_size=bs,
                            validation_data=(x_val, y_val),
                            callbacks=makeCallbacks(weights_path,monitor='val_loss',patience=3))
        
        con = True
    except Exception as e:
        print(e)
        con = False
    if con:
        # Plotting the history of the training 
        plot_history(history)
        # Write history of training to JSON file
        with open(history_path, 'w') as outfile:
            json.dump(str(history.history), outfile)
        # CM DataSet
        # Evaluate on first weights
        data = [[X, Y], [x_test, y_test]]
        model.load_weights(weights_path)
        apply_confusion_matrix(model, data)
        loss1,eval1_1,eval1_2 = compute_overall_accuracy(model,x_val, y_val, x_test, y_test)
        # Evaluate on second weights
        name = weights_path.split('.')
        weights_path_cc = name[0] + "_cc.hdf5"
        model.load_weights(weights_path_cc)
        apply_confusion_matrix(model, data)
        loss2,eval2_1,eval2_2 = compute_overall_accuracy(model,x_val, y_val, x_test, y_test)
        # exp.metric("accuracy", model.evaluate(x_test, y_test))
        # exp.end()
        #Write to result file
        file_object = open('result.txt', 'a')
        file_object.write('{}\nparameters : {}\ntrain eval : {}\ntest eval : {}\ntrain eval : {}\ntest eval : {}\n===========================\n'.format(filename,para,eval1_1,eval1_2,eval2_1,eval2_2))
        file_object.close()
        # os.system("gsutil cp -r /content/{} /content/gdrive/'My Drive'/".format(filename))
        #Go Back
        os.chdir(home_file)
        _min = (1 - max(eval1_2[1],eval2_2[1])) + min(eval1_2[0],eval2_2[0])
        file_object = open('total_result.csv', 'a')
        file_object.write('{},{},{}\n'.format(_min,para,filename))
        file_object.close()    
        return _min
    else:
        return 1000

def train(para, opt):
    # print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    return set(opt[0], opt[1], opt[2], para)