# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:18:34 2018

@author: lampa
"""

import sys, time
import random
import math
import numpy
import datetime
#import threading
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

start = datetime.datetime(2015,1,1).date()
end = datetime.datetime(2015,12,31).date()
#start = datetime.datetime(2014,1,1).date()
#end = datetime.datetime(2014,12,31).date()
#---Load Data---

#---Parameter Range---
#Parameters set by user
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

#Derived parameter attributes
rangeKDJ        = maxKDJ        - minKDJ        + 1

rangeMACD_short = maxMACD_short - minMACD_short + 1
minMACD_long    = minMACD_long  - minMACD_short
maxMACD_long    = maxMACD_long  - maxMACD_short
rangeMACD_long  = maxMACD_long  - minMACD_long  + 1
rangeMACD_hist  = maxMACD_hist  - minMACD_hist  + 1

rangeEMA0       = maxEMA0       - minEMA0       + 1
minEMA1         = minEMA1       - minEMA0
maxEMA1         = maxEMA1       - maxEMA0
rangeEMA1       = maxEMA1       - minEMA1       + 1
minEMA2         = minEMA2       - minEMA1
maxEMA2         = maxEMA2       - maxEMA1
rangeEMA2       = maxEMA2       - minEMA2       + 1
minEMA3         = minEMA3       - minEMA2
maxEMA3         = maxEMA3       - maxEMA2
rangeEMA3       = maxEMA3       - minEMA3       + 1

rangeVMA0       = maxVMA0       - minVMA0       + 1
minVMA1         = minVMA1       - minVMA0
maxVMA1         = maxVMA1       - maxVMA0
rangeVMA1       = maxVMA1       - minVMA1       + 1

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
        #input
        self.start = start
        self.end = end
        self.macd = 8, 30, 9
        self.kdj = 9, 9, 9
        self.ema = 10, 30, 60, 90
        self.vma = 5, 10
        #output
        self.profit = 0
        self.tradeCount = 0
        self.drawDown = 0


#Parameter n is added for the sake of DEAP.base.__init__
def GenIndividual(n=1):
    individual = [random.uniform(0, 1),#KDJ_K
                  random.uniform(0, 1),#KDJ_D
                  random.uniform(0, 1),#KDJ_J
                  random.uniform(0, 1),#MACD_short
                  random.uniform(0, 1),#MACD_long
                  random.uniform(0, 1),#MACD_hist
                  random.uniform(0, 1),#EMA_0
                  random.uniform(0, 1),#EMA_1
                  random.uniform(0, 1),#EMA_2
                  random.uniform(0, 1),#EMA_3
                  random.uniform(0, 1),#VMA_0
                  random.uniform(0, 1)]#vma_1
    return individual


# Convert to real indicator honoring the constraints
def Genome2Indicator(individual):
    parms = IndicatorParms(start=start, end=end)
    
    parms.kdj   =  (math.floor(minKDJ + rangeKDJ * individual[0]), 
                    math.floor(minKDJ + rangeKDJ * individual[1]), 
                    math.floor(minKDJ + rangeKDJ * individual[2]))
    
    x0 = math.floor(minMACD_short + rangeMACD_short * individual[3])
    x1 = math.floor(minMACD_long  + rangeMACD_long  * individual[4])
    parms.macd  =  (x0, 
                    x0 + x1,
                    math.floor(minMACD_hist + rangeMACD_hist * individual[5]))
                    
    x0 = math.floor(minEMA0 + rangeEMA0 * individual[6])
    x1 = math.floor(minEMA1 + rangeEMA1 * individual[7])
    x2 = math.floor(minEMA2 + rangeEMA2 * individual[8])
    x3 = math.floor(minEMA3 + rangeEMA3 * individual[9])
    parms.ema = (x0, 
                 x0 + x1,
                 x0 + x1 + x2,
                 x0 + x1 + x2 + x3) 

    x0 = math.floor(minVMA0 + rangeVMA0 * individual[10])
    x1 = math.floor(minVMA1 + rangeVMA1 * individual[11])             
    parms.vma = (x0, 
                 x0 + x1)
    return parms


def Profit(f, individual):
    parms = Genome2Indicator(individual)
    
    f, parms = optparms.Profit(f, parms)
#    parms.profit = sum(parms.macd)
#    parms.profit = sum(parms.kdj) + parms.profit
#    parms.profit = sum(parms.ema) + parms.profit
#    parms.profit = sum(parms.vma) + parms.profit
    return parms.profit,#Need to return a tuple


#https://stackoverflow.com/questions/1212779/detecting-when-a-python-script-is-being-run-interactively-in-ipython
def InIPython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


#---Configure GA---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, GenIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,)

toolbox.register("evaluate", Profit, f)
toolbox.register("mate", tools.cxTwoPoint)
#https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutShuffleIndexes
# toolbox.register("mutate", tools.mutUniformInt, indpb=0.2, low=2, up=30)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, indpb=0.2, low=0, up=1)
toolbox.register("select", tools.selTournament, tournsize=3)
#---Configure GA---
    
    
N_TOPS = 5
def main(popSize=10, nIterations=5):
    random.seed(64)
    
    pop = toolbox.population(n=popSize)

    hof = tools.HallOfFame(N_TOPS)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=nIterations, 
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof


if __name__ == "__main__":
    if not InIPython():#Do not register multiprocessing in interactive python
        #https://www.programcreek.com/python/example/13158/multiprocessing.freeze_support
        if sys.platform.startswith('win'):
            # Hack for multiprocessing.freeze_support() to work from a
            # setuptools-generated entry point.
            multiprocessing.freeze_support()

    #---Run GA---
    if not InIPython():#Do not register multiprocessing in interactive python
        #http://deap.readthedocs.io/en/master/tutorials/basic/part4.html
        #Warning As stated in the multiprocessing guidelines, under Windows, 
        #a process pool must be protected in a if __name__ == "__main__" 
        #section because of the way processes are initialized.
        pool = multiprocessing.Pool(processes=int(sys.argv[1]))
        toolbox.register("map", pool.map)


    print('\nStart optimization...\n')
    elapsedTime = time.time()

    pop, log, hofs = main(popSize=10, nIterations=1)
    
    elapsedTime = time.time() - elapsedTime
    print('\nOptimization finished in %f seconds' % elapsedTime)
    #---Run GA---
    
    #---Report Result---    
    print('\n               (KDJ)           (MACD)          (EMA)                (VMA)        TradeCount MaxDrawDown(%) Profit')
    defaultParms = IndicatorParms(start=start, end=end)
    f, defaultParms = optparms.Profit(f, defaultParms, logLevel=1)
    print('Default parms: (%3d, %3d, %3d) (%3d, %3d, %3d) (%3d, %3d, %3d, %3d) (%3d, %3d) %3d      %8.2f %8.2f' % 
          (defaultParms.kdj[0], defaultParms.kdj[1], defaultParms.kdj[2], 
           defaultParms.macd[0], defaultParms.macd[1], defaultParms.macd[2], 
           defaultParms.ema[0], defaultParms.ema[1], defaultParms.ema[2], defaultParms.ema[3], 
           defaultParms.vma[0], defaultParms.vma[1], 
           defaultParms.tradeCount, defaultParms.drawDown, 
           defaultParms.profit))
    
    for hof in hofs:
        parms = Genome2Indicator(hof)
        f, parms = optparms.Profit(f, parms, logLevel=1)
        print('Top performer: (%3d, %3d, %3d) (%3d, %3d, %3d) (%3d, %3d, %3d, %3d) (%3d, %3d) %3d      %8.2f %8.2f' % 
              (parms.kdj[0], parms.kdj[1], parms.kdj[2], 
               parms.macd[0], parms.macd[1], parms.macd[2], 
               parms.ema[0], parms.ema[1], parms.ema[2], parms.ema[3], 
               parms.vma[0], parms.vma[1], 
               parms.tradeCount, parms.drawDown, 
               parms.profit))
    #---Report Result---    
        
    
    
#                (KDJ)       (MACD)       (VMA)    (EMA)           Profit
#Default parms:  (9, 9, 9) (8, 30, 9) (5, 10) (10, 30, 60, 90) 927.819824219
#Top performer:  (19, 10, 28) (24, 35, 27) (4, 5) (24, 25, 32, 36) 1187.7199707.
#Top performer:  (9, 26, 9) (5, 16, 9) (12, 14) (15, 19, 29, 34) 1179.15991211
#Top performer:  (16, 4, 20) (27, 40, 18) (27, 28) (23, 24, 31, 34) 1179.15991211
#Top performer:  (19, 10, 28) (24, 25, 27) (27, 28) (16, 23, 30, 33) 1179.15991211
#Top performer:  (17, 19, 27) (7, 18, 9) (18, 30) (15, 19, 29, 34) 1179.15991211    
    
##https://stackoverflow.com/questions/34086112/python-multiprocessing-pool-stuck?rq=1
#from multiprocessing import Pool
#import time
#
#def f(x):
#    print x*x
#
#if __name__ == '__main__':
#    pool = Pool(processes=4)
#    pool.map(f, range(10))
#    r  = pool.map_async(f, range(10))
#    # DO STUFF
#    print 'HERE'
#    print 'MORE'
#    r.wait()
#    print 'DONE'    