import joblib
from sklearn.model_selection import cross_val_score
from model import mnist

class Measure:
  def __init__(self):
    self.model = joblib.load("mnist_5_classifier.pkl")

  def cross_val(self):
    scores = cross_val_score(
      self.model,
      mnist.X_train_scaled,
      mnist.y_train_5,
      cv=3,
      scoring="accuracy"
    )

    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())


measure_performance = Measure()
measure_performance.cross_val()