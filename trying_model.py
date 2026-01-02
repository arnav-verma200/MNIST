import joblib
import numpy as np
from PIL import Image

class try_model:
  def __init__(self):
    self.model = joblib.load("mnist_5_classifier.pkl")
    self.scaler = joblib.load("scaler.pkl")
    self.img = Image.open("testing_pics\image.png").convert("L")   # grayscale

  def predict(self):
    img = self.img.resize((28, 28))
    # Convert to array
    img_array = np.array(img)

    # Invert colors (MNIST format)
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Flatten
    img_array = img_array.reshape(1, -1)

    # Scale
    img_array_scaled = self.scaler.transform(img_array)

    # Predict
    prediction = self.model.predict(img_array_scaled)
    print("Is this a 5?", prediction[0])

try_model = try_model()
try_model.predict()