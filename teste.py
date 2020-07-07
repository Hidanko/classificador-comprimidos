from tkinter import Image
from PIL import Image, ImageOps
import tensorflow.keras
import numpy as np

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('treinamento.h5')


data = np.ndarray(shape=(1, 225, 225, 3), dtype=np.float32)
image = Image.open(r'67877-0222-05_RXNAVIMAGE10_2A3B1538.jpg')
image = image.resize((225, 225))
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)
print(prediction)