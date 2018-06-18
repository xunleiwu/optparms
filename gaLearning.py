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
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#---Load Local Modules---
# A file hosts several functions are called a module. 
# Loading a module from a file.
workPath = 'C:\\Users\\lampa\\Documents\\Github\\optparms\\'
spec = importlib.util.spec_from_file_location('optparms', workPath + 'optparms.py')
optparms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optparms)
#---Load Local Modules---

#---Load Data---
dataPath = workPath + 'data\\'
fileName = '上证指数历史数据.csv'
f = optparms.LoadTable(dataPath, fileName)

ticker = '上证指数'
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
    def __init__(self, 
                 id = 0,
                 start = datetime.datetime(1,1,1).date(), 
                 end = datetime.datetime(1,1,2).date(),
                 buyThreshold = 0.5,
                 sellThreshold = 0.5,
                 maxDrawDown = 100,#maximum allowed drawdown in %
                 kdj = (9, 9, 9),
                 macd = (8, 30, 9),
                 ema = (10, 30, 60, 90),
                 vma = (5, 10),
                 weights = [0.25, 0.25, 0.25, 0.25]):
        #input
        self.id = id
        self.start = start
        self.end = end
        #buyThreshold \in [0, 1]. When the aggregated signal >= (1-buyThreshold), then BUY.
        #Larger the threshold is, more conservative the strategy is.
        self.buyThreshold = buyThreshold
        #sellThreshold \in [0, 1]. When the aggregated signal < -(1-sellThreshold), then SELL.
        #Larger the threshold is, more aggresive the strategy is.
        self.sellThreshold = sellThreshold
        self.maxDrawDown = maxDrawDown
        #Default from Huang
        self.kdj = kdj
        self.macd = macd
        self.ema = ema
        self.vma = vma
        self.weights = weights
#        self.kdj = 11, 2, 29
#        self.macd = 3, 18, 25
#        self.ema = 13, 19, 78, 117
#        self.vma = 4, 22       
#        self.weights = [0.14, 0.14, 0.25, 0.47]
        #output
        self.profit = 0
        self.tradeCount = 0
        self.drawDown = 0


    def Print(self):
        print('Inputs:')
        print('  Id:\t\t\t%3d' % self.id)
        print('  Start Date:\t\t%s' % str(self.start))
        print('  End Date:\t\t%s' % str(self.end))
        print('  Buy Threshold:\t%.2f' % self.buyThreshold)
        print('  Sell Threshold:\t%.2f' % self.sellThreshold)
        print('  Max DrawDown (%%):\t%.2f' % self.maxDrawDown)
        print('Learned:')
        print('  KDJ:\t\t\t(%3d, %3d, %3d)' % self.kdj)
        print('  MACD:\t\t\t(%3d, %3d, %3d)' % self.macd)
        print('  EMA:\t\t\t(%3d, %3d, %3d, %3d)' % self.ema)
        print('  VMA:\t\t\t(%3d, %3d)' % self.vma)
        print('  Weights:\t\t[%.2f, %.2f, %.2f, %.2f]' % 
              (self.weights[0], self.weights[1], self.weights[2], self.weights[3]))
        print('Outputs:')
        print('  Trade Count:\t\t%d' % self.tradeCount)
        print('  DrawDown (%%):\t\t%.2f' % self.drawDown)
        print('  Profit:\t\t%.2f' % self.profit)


#tpStart = datetime.datetime(2013,1,1).date()
#tpEnd = datetime.datetime(2013,12,31).date() 
#defaultParms = IndicatorParms(start=tpStart, end=tpEnd, 
#                              buyThreshold=0.5, sellThreshold=1,
#                              maxDrawDown=100,
#                              kdj = (11, 2, 29),
#                              macd = (3, 18, 25),
#                              ema = (13, 19, 78, 117),
#                              vma = (4, 22),
#                              weights = [0.14, 0.14, 0.25, 0.47])
#f, defaultParms = optparms.Profit(f, defaultParms, logLevel=0)
#defaultParms.Print()
    
    
#Append parameter to an existing CSV file. The CSV file should have
#header row already.
def AppendParms(fileName, parms):
    with open(fileName, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', 
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([parms.id, parms.start, parms.end, 
                             parms.kdj[0], parms.kdj[1], parms.kdj[2], 
                             parms.macd[0], parms.macd[1], parms.macd[2], 
                             parms.ema[0], parms.ema[1], parms.ema[2], parms.ema[3], 
                             parms.vma[0], parms.vma[1], 
                             '{:.2f}'.format(parms.weights[0]), 
                             '{:.2f}'.format(parms.weights[1]), 
                             '{:.2f}'.format(parms.weights[2]), 
                             '{:.2f}'.format(parms.weights[3]), 
                             parms.buyThreshold, parms.sellThreshold, 
                             parms.tradeCount, 
                             '{:.2f}'.format(parms.drawDown), 
                             '{:.2f}'.format(parms.profit)])
        
        
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
                  random.uniform(0, 1),#vma_1
                  random.uniform(0, 1),#KDJ_weight
                  random.uniform(0, 1),#MACD_weight
                  random.uniform(0, 1),#EMA_weight
                  random.uniform(0, 1)]#VMA_weight
    return individual


def Profit(f, individual):
    parms = Genome2Indicator(individual)
    
    f, fFinal, parms = optparms.Profit(f, parms)
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


#---Input parameters---
#trainStart = datetime.datetime(2013,1,1).date()
#trainEnd = datetime.datetime(2016,12,31).date()
trainStart = datetime.datetime(2013,1,1).date()
trainEnd = datetime.datetime(2016,12,31).date()
buyThreshold = 0.5
sellThreshold = 1
maxDrawDown = 10#in %
#---Input parameters---


# Convert to real indicator honoring the constraints
def Genome2Indicator(individual):
    parms = IndicatorParms(start = trainStart, 
                           end = trainEnd, 
                           buyThreshold = buyThreshold,
                           sellThreshold = sellThreshold,
                           maxDrawDown = maxDrawDown)
    
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
    
    #Normalize weights
    x0 = individual[12] + individual[13] + individual[14] + individual[15]
    if x0 > 0:
        x0 = 1.0 / x0
        parms.weights = [individual[12] * x0, individual[13] * x0, individual[14] * x0, individual[15] * x0]
    else:
        parms.weights = [0.25, 0.25, 0.25, 0.25]
        
    return parms


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
    
    
N_TOPS = 2
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
        pool = multiprocessing.Pool(processes=int(sys.argv[3]))
        toolbox.register("map", pool.map)


    print('\nStart optimization...\n')
    elapsedTime = time.time()

    pop, log, hofs = main(popSize=int(sys.argv[1]), nIterations=int(sys.argv[2]))
    
    elapsedTime = time.time() - elapsedTime
    print('\nOptimization finished in %f seconds' % elapsedTime)
    #---Run GA---
    
    #---Report Result--- 
    #Generate a new CSV file and write out its header.
    fileName = 'gaLearning.csv'
    with open(fileName, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', 
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #Header
        spamwriter.writerow(['Parm Id'] + ['Start Date'] + ['End Date'] + 
                            ['KDJ Close'] + ['KDJ Low'] + ['KDJ High'] + 
                            ['MACD Short'] + ['MACD Long'] + ['MACD Histogram'] + 
                            ['EMA0'] + ['EMA1'] + ['EMA2'] + ['EMA3'] + 
                            ['VMA0'] + ['VMA1'] + 
                            ['KDJ Weight'] + ['MACD Weight'] + ['EMA Weight'] + ['VMA Weight'] + 
                            ['Buy Threshold'] + ['Sell Threshold'] +
                            ['Trade Count'] + ['Max DrawDown (%%)'] + 
                            ['Profit'])
#    print('\nResult from default parameters:')
    defaultParms = IndicatorParms(id=0, start=trainStart, end=trainEnd, 
                                  buyThreshold=1, sellThreshold=1, 
                                  maxDrawDown=100)
    f, fFinal, defaultParms = optparms.Profit(f, defaultParms, logLevel=0)
#    defaultParms.Print()
    AppendParms(fileName, defaultParms)  
    
#    print('\nResult from optimized parameters:')
    id = 1
    for hof in hofs:
        parms = Genome2Indicator(hof)
        f, fFinal, parms = optparms.Profit(f, parms, logLevel=0)
        parms.id = id 
#        parms.Print()
        AppendParms(fileName, parms)  

        optparms.SaveSignal(fFinal, ('gaLeaningSignal' + str(id) + '.csv'))
        
#        print('Top performer: (%3d, %3d, %3d) (%3d, %3d, %3d) (%3d, %3d, %3d, %3d) (%3d, %3d) (%.2f, %.2f, %.2f, %.2f)        %3d       %8.2f %8.2f' % 
#              (parms.kdj[0], parms.kdj[1], parms.kdj[2], 
#               parms.macd[0], parms.macd[1], parms.macd[2], 
#               parms.ema[0], parms.ema[1], parms.ema[2], parms.ema[3], 
#               parms.vma[0], parms.vma[1], 
#               parms.weights[0], parms.weights[1], parms.weights[2], parms.weights[3], 
#               parms.tradeCount, parms.drawDown, 
#               parms.profit))
        id = id + 1
    #---Report Result---    

    #---Scoring---
#    starts = [datetime.datetime(2013,1,1).date(),
#              datetime.datetime(2014,1,1).date(),
#              datetime.datetime(2015,1,1).date(),
#              datetime.datetime(2016,1,1).date(),
#              datetime.datetime(2017,1,1).date()
#              ]
#    
#    ends   = [datetime.datetime(2013,12,31).date(),
#              datetime.datetime(2014,12,31).date(),
#              datetime.datetime(2015,12,31).date(),
#              datetime.datetime(2016,12,31).date(),
#              datetime.datetime(2017,12,31).date()
#              ]
    
    starts = [datetime.datetime(2013,1,1).date()]
    
    ends   = [datetime.datetime(2017,12,31).date()]

    for i in range(len(starts)):
#        print('\nResult from default parameters:')
        defaultParms = IndicatorParms(id = 0, start = starts[i], end = ends[i], 
                                      buyThreshold = 1, sellThreshold = 1,
                                      maxDrawDown = 100,
                                      kdj = (9, 9, 9),
                                      macd = (8, 30, 9),
                                      ema = (10, 30, 60, 90),
                                      vma = (5, 10),
                                      weights = [0.25, 0.25, 0.25, 0.25])
        f, fFinal, defaultParms = optparms.Profit(f, defaultParms, logLevel=0)
        AppendParms(fileName, defaultParms) 
#        defaultParms.Print()
    
#        print('\nResult from optimized parameters:')
        id = 1
        for hof in hofs:
            parms = Genome2Indicator(hof)
            parms.id = id
            parms.start = starts[i]
            parms.end = ends[i]
            f, fFinal, parms = optparms.Profit(f, parms, logLevel=0)
            AppendParms(fileName, parms) 
            #parms.Print()
            id = id + 1
    #---Scoring---
    
    print('Results are recorded in %s.' % fileName)
        
    
#          Date Range           (KDJ)          (MACD)                (EMA)      (VMA) TradeCount MaxDrawDown(%)   Profit
# 2013-01-01 2016-12-31 (  9,   9,   9) (  8,  30,   9) ( 10,  30,  60,  90) (  5,  10)          7       57.78      507.87
# 2013-01-01 2016-12-31 (  5,  26,  14) ( 22,  25,  21) (  2,   4,  13,  17) ( 18,  25)         28       33.24     1671.09

# 2013-01-01 2013-12-31 (  9,   9,   9) (  8,  30,   9) ( 10,  30,  60,  90) (  5,  10)          3       16.09     -432.19
# 2013-01-01 2013-12-31 (  5,  26,  14) ( 22,  25,  21) (  2,   4,  13,  17) ( 18,  25)          8        7.77     -246.82

# 2014-01-01 2014-12-31 (  9,   9,   9) (  8,  30,   9) ( 10,  30,  60,  90) (  5,  10)          1       29.21      927.82
# 2014-01-01 2014-12-31 (  5,  26,  14) ( 22,  25,  21) (  2,   4,  13,  17) ( 18,  25)          7       24.23      795.53

# 2015-01-01 2015-12-31 (  9,   9,   9) (  8,  30,   9) ( 10,  30,  60,  90) (  5,  10)          2       37.87      -18.93
# 2015-01-01 2015-12-31 (  5,  26,  14) ( 22,  25,  21) (  2,   4,  13,  17) ( 18,  25)          4       32.20     1126.26

# 2016-01-01 2016-12-31 (  9,   9,   9) (  8,  30,   9) ( 10,  30,  60,  90) (  5,  10)          1        6.06       19.76
# 2016-01-01 2016-12-31 (  5,  26,  14) ( 22,  25,  21) (  2,   4,  13,  17) ( 18,  25)          8        5.85       40.72
    
#https://stackoverflow.com/questions/34086112/python-multiprocessing-pool-stuck?rq=1
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

#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
#
#n = 1000
#xs = np.random.randn(n).cumsum()
#i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
#j = np.argmax(xs[:i]) # start of period
#
#plt.plot(xs)
#plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)