from gc import callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('SK/voice.csv')
# print(data)
# print(data.info())

### Encoding labels ###
x = LabelEncoder()
data['label'] = x.fit_transform(data['label'])
print(dict(enumerate(x.classes_)))
# print(data)

### Splitting and Scalling ###
y = data['label'].copy()
X = data.drop('label', axis=1).copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42)

### Modeling and Training ###

# print(X.shape)

inputs = tf.keras.Input(shape=(X.shape[1],))
a = tf.keras.layers.Dense(64, activation='relu')(inputs)
a = tf.keras.layers.Dense(64, activation='relu')(a)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(a)

model = tf.keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

print(model.evaluate(X_test, y_test))
