# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:20:19 2019

@author: kirthi
"""
#%%
import numpy as np
import argparse
import os, sys
import random
from common import *

#%% - State Table class
class BoardConfiguration:
    def initializeBoxConfigs(self):
        numLinesPerRow = self._boardType
        numCols = self._boardType + 1
        self._boxConfigs = np.zeros((self._boardType**2,1),'uint')
        for ind1 in range(self._boxConfigs.shape[0]):
            line1 = 1<<(ind1)
            line2 = 1<<(ind1+numLinesPerRow)
            line3 = 1<<(ind1+(numLinesPerRow*numCols)+int(ind1/numLinesPerRow))
            line4 = 1<<(ind1+(numLinesPerRow*numCols)+int(ind1/numLinesPerRow)+1)
            self._boxConfigs[ind1,0] = (line1 | line2 | line3 | line4)
            print_debug("Box Config:",bin(self._boxConfigs[ind1,0]))

    def initializePossibleActions(self):
        numLines = 2*(self._boardType+1)*self._boardType
        self._numStates = 2**numLines

        self._possibleActions = [None]*self._numStates
        for ind1 in range(self._numStates):
#            print_debug(ind1)
            data = 2**numLines- 1
            data ^= ind1
            possActions = list()
            actionNum = 1
            while(data != 0):
                if((data & 1 == 1)):
                    possActions.append(actionNum)
                actionNum += 1
                data = data >> 1
            self._possibleActions[ind1] = (ind1, possActions)
#            print_debug("PossibleActions:",bin(self._possibleActions[ind1][0]),self._possibleActions[ind1][1])

    def getPossibleActions(self):
        return self._possibleActions

    def getBoardState(self):
        return self._boardState

    def setBoardState(self, boardVal):
        self._boardState = boardVal

    def stateTransition(self, action):
        numOfBoxesFilled = 0
        print_debug("Current Board State",bin(self._boardState))
        boxState = (self._boxConfigs & self._boardState == self._boxConfigs)
        print_debug("BoxState",boxState)
        prevUnfilledBoxes = self._boxConfigs[boxState == False]
        print_debug("Unfilled Box State",prevUnfilledBoxes)

        action = 1 << (action-1)
        newState = self._boardState | action
        print_debug("New State:",newState)
        newBoxState = (prevUnfilledBoxes & newState == prevUnfilledBoxes)
        print_debug("New Box State",newBoxState)

        numOfBoxesFilled = np.sum(newBoxState)
        print_debug("Num Of Boxes Filled",numOfBoxesFilled)

        self._boardState = newState
        print_debug("New Board State:",bin(self._boardState))

        return numOfBoxesFilled


    def wholeBoardFilled(self):
        numLines = 2*(self._boardType+1)*self._boardType
        if(self._boardState == (2**(numLines)-1)):
            return True
        else:
            return False

    def getStateActionCount(self):
        numActions = 2*self._boardType*(self._boardType+1)
        numStates = 2**numActions
        return (numActions, numStates)

    def __init__(self , boardType):
        self._boardType = boardType
        self._boardState = 0
        self.initializeBoxConfigs()
        self.initializePossibleActions()

    boardState = property(fget = getBoardState , fset = setBoardState)

#    hashTable = property(fget = get_hashTable)

