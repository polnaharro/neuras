import torch
import numpy as np
from datetime import datetime as dt

from util import *
from mar import *

from torch.nn.functional import nll_loss
from torch.optim import SGD

class ModPT():
    """
        Clase empleada para el modelado acústico basado en redes neuronales del estilo
        de PyTorch y compatible con las funciones 'entorch()' y 'recorch()'.

        Los objetos de la clase (modelos) pueden leerse de fichero, con el argumento
        'ficMod', o inicializarse desde cero, usando el argumento 'ficLisUni' para
        obtener la lista de unidades a modelar y/o reconocer.

        Tanto la función de pérdidas como el optimizador pueden especificarse en la
        invocación: la función de coste con el argumento 'funcLoss', que por defecto
        es igual a 'nll_loss'; y el optimizador con el argumento 'Optim', que por
        defecto es la clase 'SGD' con el paso de aprendizaje *congelado* a
        'lr=1.e-5'.
    """

    def __init__(self, ficLisUni=None, ficMod=None, red=None,
                 funcLoss=nll_loss, Optim=lambda params: SGD(params, lr=1.e-5)):
        if ficMod:
            self.leeMod(ficMod)
        else:
            unidades = leeLis(ficLisUni)

            self.red = red
            self.red.unidades = unidades
        
        self.resultado = []
        
        self.funcLoss = funcLoss
        self.optim = Optim(self.red.parameters())
        
        self.optim_history = []


    def escrMod(self, ficMod):
        try:
            chkPathName(ficMod)
            torch.jit.script(self.red).save(ficMod)
        except OSError as err:
            raise Exception(f'No se puede escribir el modelo {ficMod}: {err.strerror}')
        except:
            raise Exception(f'No se puede escribir el modelo {ficMod}')
    
    def leeMod(self, ficMod):
        try:
            self.red = torch.jit.load(ficMod)
        except OSError as err:
            raise Exception(f'No se puede leer el modelo {ficMod}: {err.strerror}')
        except:
            raise Exception(f'No se puede leer el modelo {ficMod}')
    
    def inicEntr(self):
        self.optim.zero_grad()
    
    def __add__(self, señal):
        salida = self.red(señal.prm).swapdims(1,2)
        loss = self.funcLoss(salida, señal.trn)
        loss.backward()

        return self
    
    def recaMod(self):
        self.optim.step()
    
    def __call__(self, señal):
        return self.red.unidades[self.red(señal.prm).argmax()]
    
    def inicEval(self):
        self.loss = 0
        self.numUni = 0
        self.corr = 0.

    def addEval(self, señal):
        salida = self.red(señal.prm).swapdims(1,2)

        self.loss += self.funcLoss(salida, señal.trn).item()
        self.numUni += 1.

        self.corr += self(señal) == self.red.unidades[señal.trn.squeeze()]
        
    def recaEval(self):
        self.loss /= self.numUni
        self.corr /= self.numUni
        self.resultado.append(self.corr )

    def printEval(self, epo):
        print(f'{epo=}\t{self.loss=}\t{self.corr=:.2%}\t({dt.now():%d/%b/%y %H:%M:%S})\n')

def calcDimIni(dirPrm, *ficLisPrm):
    """
        Función de conveniencia para determinar el número de coeficientes de las
        señales parametrizadas en el directorio 'dirPrm'. Es útil para dimensionar
        adecuadamente la capa de entrada de una red neuronal.
    """

    pathPrm = pathName(dirPrm, leeLis(*ficLisPrm)[0], 'prm')
    return len(np.load(pathPrm))
    

def calcDimSal(ficLisUni):
    """
        Función de conveniencia para determinar el número de unidades de una lista de
        unidades acústicas. Es útil para dimensionar adecuadamente la capa de salida
        de una red neuronal.
    """
    
    return len(leeLis(ficLisUni))


from collections import namedtuple

def lotesPT(dirPrm, dirMar, ficLisUni, *ficLisSen):
    """
        Función que proporciona lotes de señales compatibles con las funciones 'entorch()'
        y 'recorch()', y con las redes neuronales de PyTorch.

        En esta versión, todo el material indicado por las listas de señales '*ficLisSen'
        es incluido en un único lote.

        Cada señal está representada por una tupla nominada (namedtuple) en la que se
        tienen los campos siguientes:

        prm: señal parametrizada con un formato compatible con el admitido por las
             redes definidas en TorchAudio (como DeepSpeech o Wav2Vec): BxCxTxF, donde
             B es el minilote, C es el canal, T es el tiempo y F es la feature. Este
             formato también es compatible con las redes definidas en neuras/mlp.py.
             
             Dado que tanto el tamaño del minilote, como el número de canales, como la
             duración de las señales usadas en el reconocimiento de vocales es uno,
             las dimensiones que tendrá la señal parametrizada es 1x1x1xF, donde F
             es el número de coeficientes de la señal parametrizada.-
             
        trn: transcripción de la señal con un formato compatible con el admitido por
             las funciones de coste definidas en torch.nn.functional (como nll_loss):
             BxT, donde B es el minilote y T es el tiempo. La transcripción en sí misma
             es el índice de la unidad en la lista de unidades.
             
             Dado que tanto el tamaño del minilote como la duración de la señal son
             igual es a uno, la dimensión de la transcripción ha de ser 1x1.

        sen: Nombre de la señal. Sólo se usa en el reconocimiento para saber el nombre
             del fichero en el que se ha de escribir el resultado.
    """

    unidades = leeLis(ficLisUni)
    señal = namedtuple('señal', ['sen', 'prm', 'trn'])
    lote = []
    for sen in leeLis(*ficLisSen):
        pathPrm = pathName(dirPrm, sen, 'prm')
        prm = torch.tensor(np.load(pathPrm), dtype=torch.float).reshape(1, 1, 1, -1)

        if dirMar:
            pathMar = pathName(dirMar, sen, 'mar')
            uni = cogeTrn(pathMar)
            trn=torch.tensor([[unidades.index(uni)]])
        else:
            trn = None

        lote.append(señal(sen=sen, prm=prm, trn=trn))
    
    return [lote]
