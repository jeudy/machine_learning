#!/usr/bin/python
# -*- coding: utf-8 -*-


from sklearn import datasets
import time
import matplotlib.pyplot as plt
from matplotlib import animation


#Load the digits dataset
digits = datasets.load_digits()

fig = plt.figure("Visualizacion de Digitos")
ax = fig.add_subplot(111, title="Digitos")

# La función plot devuelve 2 objetos, por eso la , en la asignación

sp = ax.imshow(digits.images[0], cmap=plt.cm.gray_r)

def update(i):
    sp.set_data(digits.images[i])
    ax.set_title('Digito: %d' % (digits.target[i]))
    return sp,

ani = animation.FuncAnimation(fig, update, frames=len(digits.data), interval=1000, repeat=False)

#plt.figure("Digito %s" % digits.target[-2])
#plt.imshow(digits.images[-2], cmap=plt.cm.gray_r)
plt.show()
