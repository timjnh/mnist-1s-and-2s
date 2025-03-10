import io
import duckdb
import numpy
from PIL import Image
from dataclasses import dataclass
from enum import IntEnum
from typing import List
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

class Label(IntEnum):
  ONE = 0
  TWO = 1

@dataclass
class Example:
  bytes: List[int]
  label: Label

def get_data_for_row(image, label) -> Example:
  image = Image.open(io.BytesIO(image["bytes"]))
  return Example(
    [pixel for pixel in image.getdata()],
    Label.ONE if label == 1 else Label.TWO
  )

# https://huggingface.co/datasets/ylecun/mnist
duckdb.sql("CREATE VIEW train AS (SELECT * FROM read_parquet('data/train-00000-of-00001.parquet'))")
query = duckdb.sql("SELECT * FROM train WHERE label = '1' OR label = '2'")

X_train = []
Y_train = []

while True:
  rows = query.fetchmany(10)
  if not rows:
    break
  for image, label in rows:
    example = get_data_for_row(image, label)
    
    X_train.append(example.bytes)
    Y_train.append([example.label])

print(f"Loaded {len(X_train)} training examples")

model = Sequential([
  Dense(units=28*28, activation='relu'),
  Dense(units=25, activation='relu'),
  Dense(units=1, activation='sigmoid')
])

model.compile(
  loss=BinaryCrossentropy(),
  optimizer=Adam(learning_rate=1e-4)
)

model.fit(
  numpy.array(X_train, dtype=numpy.uint8),
  numpy.array(Y_train, dtype=numpy.uint8),
  epochs=100
)

duckdb.sql("CREATE VIEW test AS (SELECT * FROM read_parquet('data/test-00000-of-00001.parquet'))")
query = duckdb.sql("SELECT image, label FROM test WHERE label = '1' OR label = '2'")

incorrect_count = 0
while True:
  rows = query.fetchmany(10)
  if not rows:
    break
  for image, label in rows:
    example = get_data_for_row(image, label)

    prediction = model.predict(numpy.array([example.bytes], dtype=numpy.uint8), verbose=0)
    label_prediction = 1 if prediction < 0.5 else 2
    print(f"Prediction: {label_prediction}, Actual: {example.label + 1}")

    if label_prediction != example.label + 1:
      values = numpy.array(example.bytes, dtype=numpy.uint8).reshape(28,-1)
      image = Image.fromarray(values, 'L')

      incorrect_count += 1
      image.save(f'incorrect-{example.label + 1}-{incorrect_count}.png')

print("Done!")