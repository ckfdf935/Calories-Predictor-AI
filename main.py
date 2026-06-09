import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import numpy as np

file_id = "file_id"
url = f"https://drive.google.com/uc?id={file_id}"
name_file = "dataU.csv"

gdown.download(url, name_file, quiet=False)


with open(name_file, 'r', encoding='utf-8') as f:
   text = f.read()

lines = text.strip().split('\n')

p_data = [i.replace('"', '').split('\t') for i in lines]

data = pd.DataFrame(p_data[1:], columns=p_data[0])

data.columns = ['Food', 'Proteins', 'Fats', 'Carbs', 'Calories']

for col in ['Proteins', 'Fats', 'Carbs', 'Calories']:
  data[col] = data[col].str.replace(',', '.').astype(float)


x = data[['Proteins', 'Fats', 'Carbs']]
y = data[["Calories"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

s_x = StandardScaler()
s_y = StandardScaler()

x_train_s = s_x.fit_transform(x_train)
x_test_s = s_x.transform(x_test)

y_train_s = s_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_s = s_y.transform(y_test.values.reshape(-1, 1)).flatten()


model = Sequential([
  Dense(3, activation='relu', input_shape=(x_train_s.shape[1],)),
  BatchNormalization(),
  Dropout(0.2),

  Dense(32, activation='relu'),
  BatchNormalization(),
  Dropout(0.2),

  Dense(16, activation='relu'),
  BatchNormalization(),
  Dropout(0.2),

  Dense(1, activation='relu')
])


model.compile(optimizer='adam',
              loss='mse',
              metrics = ['mae']
             )

history = model.fit(x_train_s, y_train_s,
                    validation_split=0.2,
                    epochs = 50,
                    batch_size=32,
                    verbose=1)

test_loss, test_mae = model.evaluate(x_test_s, y_test_s)
print(f"MSE на тестовых данных: {test_loss:.4f}")
print(f"MAE на тестовых данных: {test_mae:.4f}")
