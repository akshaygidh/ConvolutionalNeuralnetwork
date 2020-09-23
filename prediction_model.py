#### Tumour prediction
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import cv2
f = Path("/content/drive/My Drive/model/model _structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("/content/drive/My Drive/model/model_weights.h5")
img = image.load_img("/content/drive/My Drive/brain images/download.jpg", target_size=(128,128))
image_to_test = image.img_to_array(img)
image_to_test = image_to_test.reshape(-1,128,128,1)
list_of_images = np.expand_dims(image_to_test, axis=0)
results = model.predict(image_to_test)
single_result = results[0][0]
print(single_result)
