from sklearn import datasets, neighbors

digits = datasets.load_digits()

digits_data = digits.data
labels = digits.target

n_samples = len(digits_data)

# Divide los datos en 2 conjuntos: para entrenamiento (75% de las instancias)
# y para pruebas (25% restante)
train_dataset = digits_data[:.75 * n_samples]
train_labels = labels[:.75 * n_samples]
test_dataset = digits_data[.75 * n_samples:]
test_labels = labels[.75 * n_samples:]

# Probar otros algoritmos
# 
knn = neighbors.KNeighborsClassifier(algorithm='brute', n_neighbors=3)

knn.fit(train_dataset, train_labels)

i = 0
errores = 0

for instance in test_dataset:

    predicted = knn.predict(test_dataset[i])
    print 'Clasificando instancia. Etiqueta real: %s. Predicha: %s' % (test_labels[i], predicted)

    if predicted != test_labels[i]:
        errores += 1

    i += 1

print 'Error rate de skilearn: %s' % (errores / len(test_dataset))

