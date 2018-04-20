# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:18:34 2018

@author: lampa
"""

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import datetime

#---Parameter range---
maxKDJ = 30
maxMACD_short=30
maxMACD_long=60
maxMACD_hist=30
maxEMA0=30 
maxEMA1=60 
maxEMA2=90 
maxEMA3=120
maxVMA0=30
maxVMA1=60

#K, D, J can vary from 1 to 30
KDJ_range = range(1, maxKDJ+1)

#MACD
MACD_short_range = range(1, maxMACD_short+1)
MACD_long_range = range(1, (maxMACD_long-maxMACD_short)+1)#MACD_short + MACD_longRange[i]
MACD_hist_range = range(1, maxMACD_hist+1)

#EMA
EMA_0_range = range(1, maxEMA0+1)
EMA_1_range = range(1, (maxEMA1-maxEMA0)+1)#EMA_0 + EMA_1_range[i]
EMA_2_range = range(1, (maxEMA2-maxEMA1)+1)#EMA_1 + EMA_2_range[i]
EMA_3_range = range(1, (maxEMA3-maxEMA2)+1)#EMA_2 + EMA_3_range[i]
#VMA
VMA_0_range = range(1, maxVMA0+1)
VMA_1_range = range(1, (maxVMA1-maxVMA0)+1)#VMA_0 + VMA_1_range[i]
#---Parameter range---


# Initial parameters
class IndicatorParms:
    def __init__(self):
        self.start = datetime.datetime(1,1,1).date()
        self.end = datetime.datetime(1,1,2).date()
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


def Profit(individual):
    #Convert to real indicator honoring the constraints
    parms = IndicatorParms()
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
    
    parms.profit = sum(parms.macd)
    parms.profit = sum(parms.kdj) + parms.profit
    parms.profit = sum(parms.ema) + parms.profit
    parms.profit = sum(parms.vma) + parms.profit
    return parms.profit,#Need to return a tuple


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, GenIndividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,)

toolbox.register("evaluate", Profit)
toolbox.register("mate", tools.cxTwoPoint)
#https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutShuffleIndexes
toolbox.register("mutate", tools.mutUniformInt, indpb=0.05, low=1, up=31)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=10)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    main()
