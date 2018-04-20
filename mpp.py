# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:22:02 2018

@author: lampa
"""
#import talib
#import numpy as np
#import pandas as pd
import datetime
#import math
#from pandas_datareader import data, wb
#import pandas_datareader.data as web
#from stockstats import StockDataFrame
#import sys
#import thread
import threading
import multiprocessing

import importlib.util

# A file hosts several functions are called a module. 
# Loading a module from a file.
workPath = 'C:\\Users\\lampa\\Documents\\TrunkCover\\'
spec = importlib.util.spec_from_file_location('optparms', workPath + 'optparms.py')
optparms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optparms)

dataPath = 'C:\\Users\\lampa\\Documents\\TrunkCover\\data\\'
fileName = '上证指数历史数据.csv'
f = optparms.LoadTable(dataPath, fileName)

ticker = '上证指数'

#start = datetime.datetime(2014,1,1).date()
#end = datetime.datetime(2014,12,31).date()
start = datetime.datetime(2009,1,5).date()
end = datetime.datetime(2017,12,31).date()
# =============================================================================
# # Download data from yahoo/google finance
# ticker = "TSLA"
# # Yahoo Daily has been immediately deprecated due to large breaks in the API without the
# # introduction of a stable replacement. Pull Requests to re-enable these data
# # connectors are welcome.
# #f=web.DataReader(ticker, 'yahoo',  start, end)
# f=web.DataReader(ticker, 'google', start, end)
# #f['SMA_20'] = talib.SMA(np.asarray(f['Close']), 20)
# =============================================================================
#f['SMA_50'] = talib.SMA(np.asarray(f['Close']), 50) 
#f.plot(y= ['Close','SMA_20','SMA_50'], title=ticker+' Close & Moving Averages')


# Initial parameters
class Parms:
    def __init__(self):
        self.start = datetime.datetime(1,1,1).date()
        self.end = datetime.datetime(1,1,2).date()
        self.kdj = 9, 9, 9
        self.macd = 8, 30, 9
        self.ema = 10, 30, 60, 90
        self.vma = 5, 10
        self.profit = 0
        self.tradeCount = 0
parms = Parms()

parms.start = start
parms.end = end
#parms.kdj = 6,10,10
#parms.macd = 10,30,10
#parms.ema = 10, 30, 60, 90
#parms.vma = 1, 10
f, parms = optparms.Profit(f, parms)
#f, maxProfit = optparms.MaxProfit(f, parms)
#print('Profit, maxProfit = ', profit0, maxProfit)


def FindOptParms(results, f, KDJ_closes, KDJ_lows, KDJ_highs, 
                 MACD_fasts, MACD_slows, MACD_signals, 
                 EMA0s, EMA1s, EMA2s, EMA3s, 
                 VMA0s, VMA1s):
    nIterations = (max(KDJ_closes) - min(KDJ_closes) + 1)
    nIterations = nIterations * (max(KDJ_lows) - min(KDJ_lows) + 1)
    nIterations = nIterations * (max(KDJ_highs) - min(KDJ_highs) + 1)
    nIterations = nIterations * (max(MACD_fasts) - min(MACD_fasts) + 1)
    nIterations = nIterations * (max(MACD_slows) - min(MACD_slows) + 1)
    nIterations = nIterations * (max(MACD_signals) - min(MACD_signals) + 1)
    nIterations = nIterations * (max(EMA0s) - min(EMA0s) + 1)
    nIterations = nIterations * (max(EMA1s) - min(EMA1s) + 1)
    nIterations = nIterations * (max(EMA2s) - min(EMA2s) + 1)
    nIterations = nIterations * (max(EMA3s) - min(EMA3s) + 1)
    nIterations = nIterations * (max(VMA0s) - min(VMA0s) + 1)
    nIterations = nIterations * (max(VMA1s) - min(VMA1s) + 1)
    
    #parmsList = []
    
    i = 0
    maxProfit = profit0
    for KDJ_close in KDJ_closes:
        for KDJ_low in KDJ_lows:
            for KDJ_high in KDJ_highs:
                for MACD_fast in MACD_fasts:
                    for MACD_slow in MACD_slows:
                        for MACD_signal in MACD_signals:
                            for EMA0 in EMA0s:
                                for EMA1 in EMA1s:
                                    for EMA2 in EMA2s:
                                        for EMA3 in EMA3s:
                                            for VMA0 in VMA0s:
                                                for VMA1 in VMA1s:
                                                    print('\rIteration: %d of %d' % (i, nIterations), end='\r', flush=True)
                                                    i = i + 1
                                                    
                                                    parms.kdj = KDJ_close, KDJ_low, KDJ_high
                                                    parms.macd = MACD_fast, MACD_slow, MACD_signal
                                                    parms.ema = EMA0, EMA1, EMA2, EMA3
                                                    parms.vma = VMA0, VMA1
                                                    f, parms = optparms.Profit(f, parms)
                                                    
                                                    if parms.profit > maxProfit:
                                                        results.put(parms)
                                                        #parmsList.append(parms)
                                                        maxProfit = parms.profit
                                                            
#                                                        print('\n')
#                                                        print('KDJ: ', parms.kdj)
#                                                        print('MACD: ', parms.macd)
#                                                        print('EMA: ', parms.ema)
#                                                        print('VMA: ', parms.vma)
#                                                        print('Profit = ', parms.profit)                                                                                    
                    
                                                
if __name__ == "__main__":
#    KDJ_closes   = range(1, 11)
#    KDJ_lows     = range(1, 11)
#    KDJ_highs    = range(1, 11)
#    MACD_fasts   = range(2, 11)#MACD_fast cannot be smaller than 2
#    MACD_slows   = range(30, 31)
#    MACD_signals = range(1, 11)
#    EMA0s        = range(10, 11)
#    EMA1s        = range(30, 31)
#    EMA2s        = range(60, 61)
#    EMA3s        = range(90, 91)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#    VMA0s        = range(1, 11)
#    VMA1s        = range(10, 11)
    KDJ_closes   = range(1, 11)
    KDJ_lows     = range(9, 11)
    KDJ_highs    = range(9, 11)
    MACD_fasts   = range(10, 11)#MACD_fast cannot be smaller than 2
    MACD_slows   = range(30, 31)
    MACD_signals = range(8, 11)
    EMA0s        = range(10, 11)
    EMA1s        = range(30, 31)
    EMA2s        = range(60, 61)
    EMA3s        = range(90, 91)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    VMA0s        = range(5, 11)
    VMA1s        = range(10, 11)
    
    results = multiprocessing.Queue()
    
    jobs= []
    for i in KDJ_closes:
        process = multiprocessing.Process(target=FindOptParms, args=(results, f, range(i, i+1), KDJ_lows, KDJ_highs, 
                                                                     MACD_fasts, MACD_slows, MACD_signals, 
                                                                     EMA0s, EMA1s, EMA2s, EMA3s, VMA0s, VMA1s,))
        jobs.append(process)
#        process.start()
        
    # Start the threads (i.e. calculate the random number lists)
    for j in jobs:
        	j.start()
    
    	# Ensure all of the threads have finished
    for j in jobs:
        	j.join()

    print('\n')        
    #Print results
    while not results.empty():
        parms = results.get()
        
        print('KDJ: ', parms.kdj)
        print('MACD: ', parms.macd)
        print('EMA: ', parms.ema)
        print('VMA: ', parms.vma)
        print('Profit = ', parms.profit) 
        print('TradeCount = ', parms.tradeCount) 
        print('\n')              
    