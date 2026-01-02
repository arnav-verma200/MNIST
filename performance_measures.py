import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from model import mnist

class Measure:
  def __init__(self):
    self.model = joblib.load("mnist_5_classifier.pkl")

  def cross_val(self):
    scores = cross_val_score(
      self.model,
      mnist.X_test_scaled,
      mnist.y_test_5,
      cv=3,
      scoring="accuracy"
    )

    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())
    
  
  def generate_matrix(self):
    y_true = mnist.y_test_5
    y_pred = self.model.predict(mnist.X_test_scaled)
    matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", matrix)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("precision Score:", precision)
    print("Recall score", recall)
    print("F1-Score", f1)
    

measure_performance = Measure()

measure_performance.cross_val()
measure_performance.generate_matrix()