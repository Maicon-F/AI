from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load saved model
model = load_model("transfer_learning_model.h5")

# Predict on a new image
img_path = "test.jpg"   # put your image path here
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("It's a dog 🐶")
else:
    print("It's a cat 🐱")