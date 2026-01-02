from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class mnist_model:
  def __init__(self):
    self.mnist = fetch_openml("mnist_784", version=1)
    self.X = self.mnist["data"]
    self.y = self.mnist["target"].astype(int)
    self.train()

  def train(self):
    X_train, X_test = self.X[:60000], self.X[60000:]
    y_train, y_test = self.y[:60000], self.y[60000:]


    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train_scaled, y_train_5)

    joblib.dump(sgd_clf, "mnist_5_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("MNIST 5-vs-not-5 model saved")

mnist_model = mnist_model()
mnist_model.train()