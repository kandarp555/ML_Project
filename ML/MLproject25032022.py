from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from gc import callbacks
from optparse import Values
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlb
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
### Enter your csv file location here...... ###
data = pd.read_csv('SK/voice.csv')
print(data)
print(data.info())

### Encoding labels ###
x = LabelEncoder()
data['label'] = x.fit_transform(data['label'])
print(dict(enumerate(x.classes_)))
print(data)

### Splitting and Scalling ###
y = data['label'].copy()
X = data.drop('label', axis=1).copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)


X = tf.keras.preprocessing.sequence.pad_sequences(
    X, dtype=np.float, maxlen=25, padding='post')


X = X.reshape(-1, 5, 5)
X = np.expand_dims(X, axis=3)
print(X.shape)

mlb.figure(figsize=(12, 12))

for i in range(9):
    mlb.subplot(3, 3, i + 1)
    mlb.imshow(np.squeeze(X[i]))
    mlb.axis('off')
    mlb.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42)

inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2], X.shape[3]))

b = tf.keras.layers.Conv2D(16, 2, activation='relu')(inputs)
b = tf.keras.layers.MaxPooling2D()(b)

b = tf.keras.layers.Conv2D(32, 1, activation='relu')(b)
b = tf.keras.layers.MaxPooling2D()(b)


b = tf.keras.layers.Flatten()(b)

b = tf.keras.layers.Dense(64, activation='relu')(b)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(b)

model = tf.keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

r = model.evaluate(X_test, y_test)

label1 = ['loss', 'accuracy', 'auc']
explode = [0.040, 0.040, 0.040]
Values = [0.09418483823537827, 0.9579390287399292, 0.9951008558273315]
print(r)
mlb.pie(Values, labels=label1, autopct="%.2f%%", explode=explode)
mlb.show()
