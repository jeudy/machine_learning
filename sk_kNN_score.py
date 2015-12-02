import os
from utils import leer_training_set, convertir_imagen
from sklearn import neighbors

def probar_algoritmo(training_dataset_path, test_dataset_path, k):
    etiquetas, contenidos = leer_training_set(training_dataset_path)

    knn = neighbors.KNeighborsClassifier(algorithm='brute', n_neighbors=k)

    # guarda la instancia del modelo
    modelo = knn.fit(contenidos, etiquetas)

    archivos = os.listdir(test_dataset_path)

    test_data = []
    test_labels = []

    for filename in archivos:
        # print 'Procesando: %s/%s' % (dir_path, filename)
        partes = filename.split('_')
        etiqueta_real = partes[0]
        test_labels.append(etiqueta_real)

        test_instance = convertir_imagen("%s/%s" % (test_dataset_path, filename))
        test_data.append(test_instance)

    print('KNN score: %f' % modelo.score(test_data, test_labels))

if __name__ == '__main__':
    probar_algoritmo('data/trainingDigits', 'data/testDigits', 1)

