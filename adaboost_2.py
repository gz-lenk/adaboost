from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# datasets ------------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(X)
# print(y)

# splitting the data --------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fitting the model ---------------------------------
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

model = abc.fit(X_train, y_train)

# Evaluating the model ----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))