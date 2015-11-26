import os
from utils import leer_training_set, clasificar, convertir_imagen


def probar_algoritmo(training_dataset_path, test_dataset_path, k):
    etiquetas, contenidos = leer_training_set(training_dataset_path)

    archivos = os.listdir(test_dataset_path)

    errores = 0

    for filename in archivos:
        # print 'Procesando: %s/%s' % (dir_path, filename)
        partes = filename.split('_')
        etiqueta_real = partes[0]

        test_instance = convertir_imagen("%s/%s" % (test_dataset_path, filename))
        predicted_label = clasificar(test_instance, contenidos, etiquetas, k)

        if etiqueta_real != predicted_label:
            errores += 1

        print "Clasificando archivo: %s. Etiqueta real: %s - etiqueta predicha: %s" % (filename,
                                                                                       etiqueta_real,
                                                                                       predicted_label)

    print 'Total de errores: %s - Procentaje de error: %s' % (errores, float(errores) / len(archivos))

if __name__ == '__main__':
    probar_algoritmo('data/trainingDigits', 'data/testDigits', 3)

