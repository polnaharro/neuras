from functools import reduce
import numpy as np
from scipy.stats import multivariate_normal
from datetime import datetime as dt

from util import *
from mar import *

class GMM():
    def __init__(self, numMix):
        self.pesos = np.zeros(numMix)
        self.medUni = [None] * numMix
        self.varUni = [None] * numMix
        self.gauss = [None] * numMix

class ModGMM():
    def __init__(self, ficMod=None, ficLisUni=None, numMix=4):
        if ficLisUni and ficMod or not ficLisUni and not ficMod:
            raise ValueError('Debe especificarse el fichero de unidades (ficLisUno) o el modelo inicial (ficMod), y sólo uno de ellos')

        if ficMod:
            self.leeMod(ficMod)
        else:
            unidades = leeLis(ficLisUni)
            self.unidades = unidades
            self.numMix = numMix
            self.gmm = {uni: GMM(numMix) for uni in self.unidades}

        self.logProb = -np.inf
    
    def escrMod(self, ficMod):
        with open(ficMod, 'wb') as fpMod:
            np.save(fpMod, self.unidades)
            np.save(fpMod, self.numMix)
            for uni in self.unidades:
                np.save(fpMod, self.gmm[uni].pesos)
                for mix in range(self.numMix):
                    np.save(fpMod, self.gmm[uni].medUni[mix])
                    np.save(fpMod, self.gmm[uni].varUni[mix])
    
    def leeMod(self, ficMod):
        with open(ficMod, 'rb') as fpMod:
            self.unidades = np.load(fpMod)
            self.numMix = int(np.load(fpMod))
            self.gmm = {}
            for uni in self.unidades:
                self.gmm[uni] = GMM(self.numMix)
                self.gmm[uni].pesos = np.load(fpMod)
                for mix in range(self.numMix):
                    self.gmm[uni].medUni[mix] = np.load(fpMod)
                    self.gmm[uni].varUni[mix] = np.load(fpMod)
                    self.gmm[uni].gauss[mix] = multivariate_normal(mean=self.gmm[uni].medUni[mix], 
                                                                   cov=self.gmm[uni].varUni[mix],
                                                                   allow_singular=True)
    
    def inicEntr(self):
        for uni in self.unidades:
            self.gmm[uni].numPrm = np.zeros(self.numMix)
            self.gmm[uni].sumPrm = [0] * self.numMix
            self.gmm[uni].sumPrm_2 = [0] * self.numMix
    
    def __add__(self, señal):
        if not all(self.gmm[señal.trn].gauss):
            pesos = np.random.rand(self.numMix)
            pesos /= sum(pesos)
        else:
            _, pesos = self.calcPesos(señal.trn, señal.prm)

        for mix in range(self.numMix):
            self.gmm[señal.trn].numPrm[mix] += pesos[mix]
            self.gmm[señal.trn].sumPrm[mix] += pesos[mix] * señal.prm
            self.gmm[señal.trn].sumPrm_2[mix] += pesos[mix] * señal.prm ** 2

        return self
    
    def recaMod(self):
        for uni in self.unidades:
            totPrm = sum(self.gmm[uni].numPrm)
            self.gmm[uni].pesos = self.gmm[uni].numPrm / totPrm
            self.gmm[uni].gauss = [None] * self.numMix
            for mix in range(self.numMix):
                self.gmm[uni].medUni[mix] = self.gmm[uni].sumPrm[mix] / self.gmm[uni].numPrm[mix]
                self.gmm[uni].varUni[mix] = self.gmm[uni].sumPrm_2[mix] / self.gmm[uni].numPrm[mix]
                self.gmm[uni].varUni[mix] -= self.gmm[uni].medUni[mix] ** 2
                self.gmm[uni].varUni[mix] = np.clip(self.gmm[uni].varUni[mix], 1.e-24, None)
                self.gmm[uni].gauss[mix] = multivariate_normal(mean=self.gmm[uni].medUni[mix], 
                                                                cov=self.gmm[uni].varUni[mix],
                                                                allow_singular=True)
    def calcPesos(self, unidad, prm):
        lProbs = [self.gmm[unidad].gauss[mix].logpdf(prm) for mix in range(self.numMix)]
        lProbMax = max(lProbs)
        for mix in range(self.numMix):
            lProbs[mix] -= lProbMax

        pesos = np.array([self.gmm[unidad].pesos[mix] * np.exp(lProbs[mix]) for mix in range(self.numMix)])
        sumPesos = sum(pesos)
        pesos /= sumPesos

        lProb = lProbMax + np.log(sumPesos)

        return lProb, pesos
 
    def __call__(self, prm):
        lProbs = {unidad: self.calcPesos(unidad, prm)[0] for unidad in self.unidades}
        return reduce(lambda x, y: max(x, y, key=lambda unidad: lProbs[unidad]), self.unidades)

    def inicEval(self):
        self.sumLogPrb = {unidad: 0 for unidad in self.unidades}
        self.numUni = {unidad: 0 for unidad in self.unidades}
        self.corr = 0.

    def addEval(self, señal):
        self.sumLogPrb[señal.trn] += self.calcPesos(señal.trn, señal.prm)[0]
        self.numUni[señal.trn] += 1
        self.corr += self(señal.prm) == señal.trn

    def recaEval(self):
        logProb = 0
        numUni = 0
        for unidad in self.unidades:
            logProb += self.sumLogPrb[unidad]
            numUni += self.numUni[unidad]

        logProb /= numUni
        self.corr /= numUni

        self.incLogProb = logProb - self.logProb
        self.logProb = logProb

    def printEval(self, epo):
        print(f'{epo=}\t{self.logProb=:.2f}\t{self.incLogProb=:.4f}\t{self.corr=:.2%}\t({dt.now():%d/%b/%y %H:%M:%S})\n')
        