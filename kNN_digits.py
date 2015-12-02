from utils import clasificar
from sklearn import datasets

def probar_algoritmo_scikit_data(k):

    digits = datasets.load_digits()

    x_digits = digits.data
    y_digits = digits.target

    n_samples = len(x_digits)

    train_dataset = x_digits[:.9 * n_samples]
    train_labels = y_digits[:.9 * n_samples]
    test_dataset = x_digits[.9 * n_samples:]
    test_labels = y_digits[.9 * n_samples:]

    errores = 0

    i = 0

    for test_instance in test_dataset:

        etiqueta_real = test_labels[i]

        i += 1

        predicted_label = clasificar(test_instance, train_dataset, train_labels, k)

        if etiqueta_real != predicted_label:
            errores += 1

        print "Clasificando instancia. Etiqueta real: %s - etiqueta predicha: %s" % (
                                                                                     etiqueta_real,
                                                                                       predicted_label)

    print 'Total de errores: %s - Procentaje de error: %s' % (errores, float(errores) / len(test_dataset))

if __name__ == '__main__':
    probar_algoritmo_scikit_data(3)

