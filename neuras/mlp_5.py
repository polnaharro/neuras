from mod_pt import *
from mlp import mlp_N
from torch.optim import Adam
from torch.nn import ReLU, LogSoftmax, sigmoid, Hardtanh
import matplotlib.pyplot as plt

dirPrm = 'prm/cepstrum.9.5.20/'
#dirPrm = 'Prm'
dirMar = 'Sen'

guiTrain = 'Gui/train.gui'
guiDevel = 'Gui/devel.gui'
guiRec = 'Gui/eval.gui'

ficLisUni = 'Lis/vocales.lis'

lotesEnt = lotesPT(dirPrm, dirMar, ficLisUni, guiTrain)
lotesDev = lotesPT(dirPrm, dirMar, ficLisUni, guiDevel)
lotesRec = lotesPT(dirPrm, None, ficLisUni, guiDevel) # Usamos guiDeval para poder evaluar el resultado
numCof = calcDimIni(dirPrm, guiTrain)
tamVoc = calcDimSal(ficLisUni)

mlp_5 = mlp_N(3, dimIni=numCof, dimInt=512, dimSal=tamVoc, clsAct=Hardtanh(), clsSal=LogSoftmax(dim=-1))

modMLP_5 = ModPT(ficLisUni=ficLisUni, red=mlp_5, Optim=lambda params: Adam(params, lr=1.e-3))
