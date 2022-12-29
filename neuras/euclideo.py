from functools import reduce
import numpy as np
from datetime import datetime as dt

from util import *
from mar import *

class ModEucl():
    """
        Clase empleada para el modelado acústico basado en distancia y compatible con 
        las funciones 'entorch()' y 'recorch()'.

        Los objetos de la clase (modelos) pueden leerse de fichero, con el argumento
        'ficMod', o inicializarse desde cero, usando el argumento 'ficLisUni' para
        obtener la lista de unidades a modelar y/o reconocer.
    """

    def __init__(self, ficMod=None, ficLisUni=None):
        """
            Inicializa un modelo de la clase 'ModEucl' a partir de una lista de unidades,
            usando el argumento 'ficLisUni', o leyéndolo del fichero 'ficMod'. Ambas
            opciones son incompatibles entre sí, pero es necesario usar una de ellas.
        """

        if ficLisUni and ficMod or not ficLisUni and not ficMod:
            raise ValueError('Debe especificarse el fichero de unidades (ficLisUno) o el'
                             'modelo inicial (ficMod), y sólo uno de ellos')

        if ficMod:
            self.leeMod(ficMod)
        else:
            self.unidades = leeLis(ficLisUni)
    
    def escrMod(self, ficMod):
        """
            Escribe el modelo en el fichero indicado por su argumento 'ficMod'.
        """

        with open(ficMod, 'wb') as fpMod:
            np.save(fpMod, self.medUni)
    
    def leeMod(self, ficMod):
        """
            Lee el modelo contenido en el fichero indicado por su argumento 'ficmod'.
        """

        with open(ficMod, "rb") as fpMod:
            self.medUni = np.load(fpMod, allow_pickle=True).item()
            self.unidades = self.medUni.keys()
    
    def inicEntr(self):
        """
            Inicializa las estructuras necesarias para realizar el entrenamiento.
        """

        self.medUni = {unidad: 0 for unidad in self.unidades}
        self.numUni = {unidad: 0 for unidad in self.unidades}
    
    def __add__(self, señal):
        """
            Sobrecarga del operador suma que añade la información de una señal al modelo
            durante su entrenamiento.
        """

        self.medUni[señal.trn] += señal.prm
        self.numUni[señal.trn] += 1

        return self
    
    def recaMod(self):
        """
            Recalcula los parámetros del modelo a partir de la información recopilada por
            el método '__add__()'.
        """

        for unidad in self.unidades:
            if self.numUni[unidad]:
                self.medUni[unidad] /= self.numUni[unidad]
    
    def __call__(self, señal):
        """
            Sobrecarga de la llamada a función que determina el resultado de reconocer la
            señal 'prm' por el modelo.
        """

        distancias = {unidad: sum(abs(self.medUni[unidad] - señal.prm) ** 2) for unidad in self.unidades}
        return reduce(lambda x, y: min(x, y, key=lambda mod: distancias[mod]), self.unidades)

    def inicEval(self):
        """
            Inicializa la estructuras necesarias para proporcionar información acerca de la
            evolución del entrenamiento.
        """
        self.varUni = {unidad: 0 for unidad in self.unidades}
        self.numUni = {unidad: 0 for unidad in self.unidades}
        self.corr = 0.

    def addEval(self, señal):
        """
            Añade la información de una señal al cálculo de la evolución del entrenamiento.
        """
        self.varUni[señal.trn] += (señal.prm - self.medUni[señal.trn]) ** 2
        self.numUni[señal.trn] += 1
        self.corr += self(señal) == señal.trn

    def recaEval(self):
        """
            Calcula las prestaciones del modelo de cara a mostrar la evolución de su
            entrenamiento.
        """
        varianza = numUni = 0
        for unidad in self.unidades:
            varianza += sum(self.varUni[unidad])
            numUni += self.numUni[unidad]

        varianza /= numUni * len(self.varUni[unidad])
        self.sigma = varianza ** 0.5
        self.corr /= numUni

    def printEval(self, epo):
        """
            Muestra en pantalla la información acerca de la evolución del entrenamiento
            calculada con el método 'recaEval'.

            El argumento 'epo' permite concer la época correspondiente a esta información.
        """

        print(f'{epo=}\t{self.sigma=}\t{self.corr=:.2%}\t({dt.now():%d/%b/%y %H:%M:%S})\n')


from collections import namedtuple

def lotesEucl(dirPrm, dirMar, *ficLisSen):
    """
        Función que proporciona lotes de señales compatibles con las funciones 'entorch()'
        y 'recorch()', y con los modelos de la clase 'ModEucl'.

        La función devuelve un iterable con un único lote que contiene todas las señales.
        Cada señal es una namedtuple con tres elementos:

        sen: el nombre de la señal tal y como aparece en 'ficLisSen'
        prm: la señal parametrizada en formato ndarray de numpy y tal y como se lee del
             directorio 'dirPrm'
        mar: La transcripción contenida en la etiqueta "LBO:" del fichero de marcas del
             directorio 'dirMar'. Si 'dirMar' evalúa a False, mar = None

        En principio, cada tipo de modelado requiere un formato específico de los lotes.
        Por ejemplo, el modelado Euclídeo es incompatible con las redes neuronales de 
        PyTorch. Sin embargo, este mismo formato de lote es también útil en modelos de
        mezcla de gaussianas, GMM, y probablemente otros.
    """

    señal = namedtuple('señal', ['sen', 'prm', 'trn'])
    lote = []
    for sen in leeLis(*ficLisSen):
        pathPrm = pathName(dirPrm, sen, 'prm')
        prm = np.load(pathPrm)

        if dirMar:
            pathMar = pathName(dirMar, sen, 'mar')
            trn = cogeTrn(pathMar)
        else:
            trn = None

        lote.append(señal(sen=sen, prm=prm, trn=trn))
    
    return [lote]