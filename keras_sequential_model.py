import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.utils



dataset = pd.read_csv('YASBABY.csv')
dataset.pop('Unnamed: 0')
#print(len(list(dataset.columns)))


dataset = dataset[dataset['number_of_casualties']<= 10]
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#print(dataset.head(2))
#print(dataset.describe(include='all'))
Y = train_dataset.number_of_casualties
X = train_dataset.drop("number_of_casualties",axis=1)


B = test_dataset.number_of_casualties
A = test_dataset.drop("number_of_casualties",axis=1)


X = X.values
Y = Y.values
A = A.values
B = B.values
print(X.shape)
print(A.shape)


yy = keras.utils.to_categorical(Y, num_classes=11)
#print(yy)

model = Sequential()
model.add(Dense(units=1024, activation='relu', input_dim=387))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=11, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

h = model.fit(X, yy, epochs=2, batch_size=64)
model.save('my_model.h5')
plt.plot(h.history['accuracy'])
plt.plot(h.history['loss'])
plt.show()


predictions = model.predict(A)
result = []
for p in predictions:
    a = p.reshape((11,)).tolist()
    b = a.index(max(a))
    result.append(b)


print(max(result))
