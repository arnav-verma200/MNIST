from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class mnist_model:
  def __init__(self):
    self.mnist = fetch_openml("mnist_784", version=1)
    self.X = self.mnist["data"]
    self.y = self.mnist["target"].astype(int)

  def train(self):
    # Store train/test splits as instance attributes
    self.X_train, self.X_test = self.X[:60000], self.X[60000:]
    self.y_train, self.y_test = self.y[:60000], self.y[60000:]

    # Binary targets for "5 vs not-5"
    self.y_train_5 = (self.y_train == 5)
    self.y_test_5 = (self.y_test == 5)

    # Fit scaler and classifier, keep them on the instance
    self.scaler = StandardScaler()
    self.X_train_scaled = self.scaler.fit_transform(self.X_train)
    self.X_test_scaled = self.scaler.transform(self.X_test)

    self.sgd_clf = SGDClassifier(random_state=42, max_iter=2000)
    self.sgd_clf.fit(self.X_train_scaled, self.y_train_5)

    # Persist trained objects
    joblib.dump(self.sgd_clf, "mnist_5_classifier.pkl")
    joblib.dump(self.scaler, "scaler.pkl")
    print("MNIST 5-vs-not-5 model saved")


# Train the model when this file is run directly
if __name__ == "__main__":
  mnist = mnist_model()
  mnist.train()
else:
  # When imported, create a trained instance for other modules to use
  mnist = mnist_model()
  mnist.train()