__author__ = 'jeudy-work'

import numpy as np
from scipy.stats import itemfreq
import os


# http://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
def unique_count(a):
    unique, inverse = np.unique(a, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return np.vstack(( unique, count)).T


def convertir_imagen(path):
    f = open(path)

    lines = f.readlines()

    resp = np.empty(shape=1024, dtype=int)

    i = 0

    for line in lines:
        # Asume que cada linea tiene 32 caracteres
        for c in line.strip():
            if not c:
                continue
            resp[i] = ord(c)
            i += 1

    f.close()

    return resp


def calcular_distancia(v1, v2):
    return np.linalg.norm(v1 - v2)


def leer_training_set(dir_path):
    archivos = os.listdir(dir_path)

    etiquetas = []
    contenidos = []

    for filename in archivos:
        # print 'Procesando: %s/%s' % (dir_path, filename)
        partes = filename.split('_')
        etiquetas.append(partes[0])
        contenidos.append(convertir_imagen("%s/%s" % (dir_path, filename)))

    return np.array(etiquetas), contenidos


def clasificar(vector, training_data, labels, k):
    distancias = np.empty(shape=len(training_data), dtype=int)

    i = 0

    for instance in training_data:
        distancias[i] = calcular_distancia(vector, instance)
        i += 1

    # Obtengo los indices de las k menores distancias
    indices = distancias.argsort()[0:k]

    # Obtengo las etiquetas
    voting_labels = labels[indices]

    # Devuelve la etiqueta mas votada luego de contar los votos
    return unique_count(voting_labels)[0][0]