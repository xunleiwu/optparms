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


class Indicators:
    #Parameter n is added for the sake of DEAP.base.__init__
    def __init__(self, n=1):
        self.start = datetime.datetime(1,1,1).date()
        self.end = datetime.datetime(1,1,2).date()
        self.profit = 0        
#        self.macd = 8, 30, 9
#        self.kdj = 9, 9, 9
#        self.ema = 10, 30, 60, 90
#        self.vma = 5, 10
        x0 = random.choice(KDJ_range)
        x1 = random.choice(KDJ_range)
        x2 = random.choice(KDJ_range)
        self.kdj = (x0, x1, x2)
    
        x0 = random.choice(MACD_short_range)
        x1 = random.choice(MACD_long_range)
        x2 = random.choice(MACD_hist_range)
        self.macd = (x0, x1, x2)

        x0 = random.choice(EMA_0_range)
        x1 = random.choice(EMA_1_range)
        x2 = random.choice(EMA_2_range)
        x3 = random.choice(EMA_3_range)
        self.ema = (x0, x1, x2, x3)
    
        x0 = random.choice(VMA_0_range)
        x1 = random.choice(VMA_1_range)
        self.vma = (x0, x1)            
    
    #For cxTwoPoint()
    def len(self):
        return 12#3 in KDJ, 3 in MACD, 4 in EMA, 2 in VMA
    
        
    def Profit(self):
        #Convert to real indicator honoring the constraints
        self.macd = (self.macd[0], 
                     self.macd[0] + self.macd[1], 
                     self.macd[2])
        self.ema = (self.ema[0], 
                    self.ema[0] + self.ema[1], 
                    self.ema[0] + self.ema[1] + self.ema[2], 
                    self.ema[0] + self.ema[1] + self.ema[2] + self.ema[3])
        self.Vma = (self.vma[0], 
                    self.vma[0] + self.vma[1]) 
        
        self.profit = sum(self.macd)
        self.profit = sum(self.kdj) + self.profit
        self.profit = sum(self.ema) + self.profit
        self.profit = sum(self.vma) + self.profit
        return self.profit,#Need to return a tuple


#---Initialize population---
def GeneratePopulation(nItems=70):
    #Parameter range
    #K, D, J can vary from 1 to 30
    KDJ_K_range = range(1, 31)
    KDJ_D_range = range(1, 31)
    KDJ_J_range = range(1, 31)
    #MACD
    MACD_short_range = range(1, 31)
    MACD_long_range = range(1, 61)#MACD_short + MACD_longRange[i]
    MACD_hist_range = range(1, 31)
    #EMA
    EMA_0_range = range(1, 31)
    EMA_1_range = range(1, 31)#EMA_0 + EMA_1_range[i]
    EMA_2_range = range(1, 31)#EMA_1 + EMA_2_range[i]
    EMA_3_range = range(1, 31)#EMA_2 + EMA_3_range[i]
    #VMA
    VMA_0_range = range(1, 31)
    VMA_1_range = range(1, 31)#VMA_0 + VMA_1_range[i]
    
    #Prepare initial population
    #Create population containing all genome
    pop = []
    
    gaParms = IndicatorParms()
    for i in range(0, 30):
        gaParms.macd = MACD_short_range[i], MACD_long_range[i], MACD_hist_range[i]
        gaParms.kdj = KDJ_K_range[i], KDJ_D_range[i], KDJ_J_range[i]
        gaParms.ema = EMA_0_range[i], EMA_1_range[i], EMA_2_range[i], EMA_3_range[i]
        gaParms.vma = VMA_0_range[i], VMA_1_range[i]
        pop.append(gaParms)
    
    #Expand initial population with random combinations
    #pop = pop.append(toolbox.population(n=70))
    x = range(1, 31)    
    #random.choices() is introduced in Python 3.6
    KDJ_K_list = random.choices(x, k=nItems)    
    KDJ_D_list = random.choices(x, k=nItems)    
    KDJ_J_list = random.choices(x, k=nItems)    
    MACD_short_list = random.choices(x, k=nItems)    
    MACD_long_list = random.choices(x, k=nItems)    
    MACD_hist_list = random.choices(x, k=nItems)    
    EMA_0_list = random.choices(x, k=nItems)    
    EMA_1_list = random.choices(x, k=nItems)    
    EMA_2_list = random.choices(x, k=nItems)    
    EMA_3_list = random.choices(x, k=nItems)    
    VMA_0_list = random.choices(x, k=nItems)    
    VMA_1_list = random.choices(x, k=nItems)    
    for i in range(0, nItems):
        gaParms.macd = MACD_short_list[i], MACD_long_list[i], MACD_hist_list[i]
        gaParms.kdj = KDJ_K_list[i], KDJ_D_list[i], KDJ_J_list[i]
        gaParms.ema = EMA_0_list[i], EMA_1_list[i], EMA_2_list[i], EMA_3_list[i]
        gaParms.vma = VMA_0_list[i], VMA_1_list[i]
        pop.append(gaParms)

    #indicatorParms = []
    #for gaParms in pop:
    #    indicatorParms.append(gaParms.ToIndicatorParms())
    return pop
#---Initialize population---


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
creator.create("Individual", Indicators, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#toolbox.register("attr_bool", random.randrange, 1, (10+1))
# Structure initializers
#toolbox.register("individual", GenGeneParms, creator.Individual, 1)
#toolbox.register("individual", GenGeneParms)
toolbox.register("individual", tools.initRepeat, creator.Individual, Indicators, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual,)

toolbox.register("evaluate", Indicators.Profit)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
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
    
    
IND_SIZE = 5

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)


