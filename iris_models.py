from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

test_instance = iris.data[0]
test_label = iris.target[0]

train_dataset = iris.data[1:]
train_labels = iris.target[1:]

modelo1 = LogisticRegression().fit(train_dataset, train_labels)

modelo2 = KNeighborsClassifier(algorithm='brute', n_neighbors=3).fit(train_dataset, train_labels)

print "Clase real para instancia de prueba: ", iris.target_names[0]

print "Prediccion con LogisticRegression", iris.target_names[modelo1.predict(test_instance)[0]]
print "Prediccion con kNN", iris.target_names[modelo2.predict(test_instance)[0]]

print "Probabilidades con LogisticRegression", modelo1.predict_proba(test_instance)
print "Probabilidades con kNN", modelo2.predict_proba(test_instance)
