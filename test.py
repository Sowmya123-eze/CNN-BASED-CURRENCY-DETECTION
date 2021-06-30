from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

img = image.load_img("./content/drive/MyDrive/fake_currency_final/dataset/test/real/Real_1.jpg",target_size=(224,224))

img = np.asarray(img)
# plt.imshow(img)
img = np.expand_dims(img, axis=0)
saved_model = load_model("fake_currency.h5")
p=saved_model .predict(img)
pro=np.max(p[0], axis=-1)

if pro >=1:
   print("Currency is fake")

else:
  print("Currency is real")