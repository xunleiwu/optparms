# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:18:34 2018

@author: lampa
"""

import random
import numpy
import datetime
import threading
import multiprocessing
import importlib.util

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#---Load Local Modules---
# A file hosts several functions are called a module. 
# Loading a module from a file.
workPath = 'C:\\Users\\lampa\\Documents\\TrunkCover\\'
spec = importlib.util.spec_from_file_location('optparms', workPath + 'optparms.py')
optparms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optparms)
#---Load Local Modules---

#---Load Data---
dataPath = 'C:\\Users\\lampa\\Documents\\TrunkCover\\data\\'
fileName = '上证指数历史数据.csv'
f = optparms.LoadTable(dataPath, fileName)

ticker = '上证指数'

#start = datetime.datetime(2015,1,1).date()
#end = datetime.datetime(2015,12,31).date()
start = datetime.datetime(2014,1,1).date()
end = datetime.datetime(2014,12,31).date()
#---Load Data---

#---Parameter range---
#Allowed minimal parameters
#defaultParms.kdj = (1,1,1)
#defaultParms.macd = (2,3,1)
#defaultParms.ema = (2,3,4,5)
#defaultParms.vma = (1,2)

minKDJ = 1
maxKDJ = 30

minMACD_short=2
maxMACD_short=30
minMACD_long=3
maxMACD_long=60
minMACD_hist=1
maxMACD_hist=30

minEMA0=2 
maxEMA0=30 
minEMA1=3 
maxEMA1=60 
minEMA2=4 
maxEMA2=90 
minEMA3=5
maxEMA3=120

minVMA0=1
maxVMA0=30
minVMA1=2
maxVMA1=60

#K, D, J can vary from 1 to 30
KDJ_range = range(minKDJ, maxKDJ+1)

#MACD
MACD_short_range = range(minMACD_short, maxMACD_short+1)
MACD_long_range = range((minMACD_long-minMACD_short), (maxMACD_long-maxMACD_short)+1)#MACD_short + MACD_longRange[i]
MACD_hist_range = range(minMACD_hist, maxMACD_hist+1)

#EMA
EMA_0_range = range(minEMA0, maxEMA0+1)
EMA_1_range = range((minEMA1-minEMA0), (maxEMA1-maxEMA0)+1)#EMA_0 + EMA_1_range[i]
EMA_2_range = range((minEMA2-minEMA1), (maxEMA2-maxEMA1)+1)#EMA_1 + EMA_2_range[i]
EMA_3_range = range((minEMA3-minEMA2), (maxEMA3-maxEMA2)+1)#EMA_2 + EMA_3_range[i]
#VMA
VMA_0_range = range(minVMA0, maxVMA0+1)
VMA_1_range = range((minVMA1-minVMA0), (maxVMA1-maxVMA0)+1)#VMA_0 + VMA_1_range[i]
#---Parameter range---


# Initial parameters
class IndicatorParms:
    def __init__(self, start=datetime.datetime(1,1,1).date(), end=datetime.datetime(1,1,2).date()):
        self.start = start
        self.end = end
        self.macd = 8, 30, 9
        self.kdj = 9, 9, 9
        self.ema = 10, 30, 60, 90
        self.vma = 5, 10
        self.profit = 0


#Parameter n is added for the sake of DEAP.base.__init__
def GenIndividual(n=1):
    individual = [random.choice(KDJ_range),#KDJ_K
                  random.choice(KDJ_range),#KDJ_D
                  random.choice(KDJ_range),#KDJ_J
                  random.choice(MACD_short_range),#MACD_short
                  random.choice(MACD_long_range),#MACD_long
                  random.choice(MACD_hist_range),#MACD_hist
                  random.choice(EMA_0_range),#EMA_0
                  random.choice(EMA_1_range),#EMA_1
                  random.choice(EMA_2_range),#EMA_2
                  random.choice(EMA_3_range),#EMA_3
                  random.choice(VMA_0_range),#VMA_0
                  random.choice(VMA_1_range)]#vma_1
    return individual


# Convert to real indicator honoring the constraints
def Genome2Indicator(individual):
    parms = IndicatorParms(start=start, end=end)
    parms.kdj = (individual[0], individual[1], individual[2])
    parms.macd = (individual[3], 
                  individual[3] + individual[4], 
                  individual[5])
    parms.ema = (individual[6], 
                 individual[6] + individual[7], 
                 individual[6] + individual[7] + individual[8], 
                 individual[6] + individual[7] + individual[8] + individual[9])
    parms.vma = (individual[10], 
                 individual[10] + individual[11])  
    return parms


def Profit(f, individual):
    parms = Genome2Indicator(individual)
    
    f, parms.profit = optparms.Profit(f, parms)
    
#    parms.profit = sum(parms.macd)
#    parms.profit = sum(parms.kdj) + parms.profit
#    parms.profit = sum(parms.ema) + parms.profit
#    parms.profit = sum(parms.vma) + parms.profit
    return parms.profit,#Need to return a tuple


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, GenIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,)

toolbox.register("evaluate", Profit, f)
toolbox.register("mate", tools.cxTwoPoint)
#https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutShuffleIndexes
toolbox.register("mutate", tools.mutUniformInt, indpb=0.2, low=2, up=30)
toolbox.register("select", tools.selTournament, tournsize=3)

N_TOPS = 5

def main():
    random.seed(64)
    
    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(N_TOPS)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hofs = main()
    
print('                (KDJ)       (MACD)       (VMA)    (EMA)           Profit')
defaultParms = IndicatorParms(start=start, end=end)
f, defaultParms.profit = optparms.Profit(f, defaultParms)
print('Default parms: ', defaultParms.kdj, defaultParms.macd, defaultParms.vma, defaultParms.ema, defaultParms.profit)

for hof in hofs:
    parms = Genome2Indicator(hof)
    f, parms.profit = optparms.Profit(f, parms)
    print('Top performer: ', parms.kdj, parms.macd, parms.vma, parms.ema, parms.profit)
        
    
    
#                (KDJ)       (MACD)       (VMA)    (EMA)           Profit
#Default parms:  (9, 9, 9) (8, 30, 9) (5, 10) (10, 30, 60, 90) 927.819824219
#Top performer:  (19, 10, 28) (24, 35, 27) (4, 5) (24, 25, 32, 36) 1187.7199707
#Top performer:  (9, 26, 9) (5, 16, 9) (12, 14) (15, 19, 29, 34) 1179.15991211
#Top performer:  (16, 4, 20) (27, 40, 18) (27, 28) (23, 24, 31, 34) 1179.15991211
#Top performer:  (19, 10, 28) (24, 25, 27) (27, 28) (16, 23, 30, 33) 1179.15991211
#Top performer:  (17, 19, 27) (7, 18, 9) (18, 30) (15, 19, 29, 34) 1179.15991211    