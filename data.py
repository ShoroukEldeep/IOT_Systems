
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

def get_data(length):
    df = pd.read_csv('Train_Test_IoT_Fridge.csv')
    df['h'] = df.time.apply(lambda x : x.split(':')[0])
    df['m'] = df.time.apply(lambda x : x.split(':')[1])
    df['s'] = df.time.apply(lambda x : x.split(':')[2])

    def temp_change(x):
        if 'low' in x:
            return 0
        else:
            return 1

    df.temp_condition = df.temp_condition.apply(temp_change)
    df.set_index(keys=['date','h'], drop=False,inplace=True)
    fdf = df.drop(['ts','date','time','h','m','s'], axis=1)

    x = []
    y = []
    for index in tqdm(df.index.unique()):
        arr = [[0,0] for i in range(0,length)]
        sub = fdf.loc[index]
        for i in range(1,sub.shape[0]): #
            arr = arr[1:]
            l1 = sub.drop(['label','type'],axis=1).values[i].tolist()
            l2 = sub.drop(['label','type'],axis=1).values[i-1].tolist()
            arr.append([l1[0] - l2[0],l1[1]])
            z = arr
            x.append(z)
            y.append(sub.drop(['fridge_temperature','temp_condition','label'],axis=1).values[i][0])

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(y)).tolist()
    encoded = tf.keras.utils.to_categorical(integer_encoded,7).tolist()

    x_train,x_test,y_train,y_test = train_test_split(x,integer_encoded,shuffle=True,random_state=42,train_size=0.9)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,shuffle=True,random_state=42,train_size=0.9)
    return x_train,x_val,x_test,y_train,y_val,y_test
