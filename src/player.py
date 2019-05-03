#%%
import numpy as np
import argparse
import os, sys
import random
import common
from boardConfiguration import *

#%%
class Player:
    def __init__(self , boardType):
        self._playerScore = 0
        self._playerWon = False
        self._numBoxesForWin = np.ceil(((boardType**2)/2)+0.1)

    def getWinFlag(self):
        return self._playerWon

    def setWinFlag(self , winFlagVal):
        self._playerWon = winFlagVal

    def getScore(self):
        return self._playerScore

    def setScore(self, scoreVal):
        self._playerScore = scoreVal

    wonFlag = property(fget = getWinFlag, fset = setWinFlag)
    score = property(fget = getScore , fset = setScore)
#%%
class RandomPlayer(Player):
    def __init__(self, boardType):
        Player.__init__(self,boardType)

    def drawLine(self, boardConfiguration):
        currentState = boardConfiguration._boardState

        possibleActionsCurrentState = boardConfiguration._possibleActions[currentState][1]
        print_debug("Possible Actions Current",possibleActionsCurrentState)
        randSeed = random.randint(0 ,  len(possibleActionsCurrentState)-1)
        print_debug("Rand Number:",randSeed)

        action = possibleActionsCurrentState[randSeed]
        print_debug("Random Action Chosen",action)
        numBoxesFilled = boardConfiguration.stateTransition(action)

        self._playerScore += numBoxesFilled

        if(self._playerScore >= self._numBoxesForWin):
            self._playerWon = True

        return numBoxesFilled


#%%
class QPlayer(Player):
    def __init__(self , boardType, learnRate, epsilonVal, discRate, qTable):
        Player.__init__(self , boardType)
        self._learningRate = learnRate
        self._discountRate = discRate
        self._epsilon = epsilonVal
        # Mode 1 refers to training and mode 2 refers to test
        self._mode = 1
        # Creating the Q table
#        self._qtable = qTable
        QPlayer._qtable = qTable

    def drawLine(self, boardConfiguration):
        currentState = boardConfiguration._boardState
        
        possActions = boardConfiguration._possibleActions[currentState][1]

        maxQVal = np.max(QPlayer._qtable[currentState,:])
        print_debug("Max QVal:",maxQVal)

        exploreRandSeed = random.uniform(0,1)
        
        actionSpace = list()
        print_debug("Poss Actions: ",possActions)
        if((exploreRandSeed < self._epsilon) and (self._mode == 1)):
            actionSpace = possActions
        else:
            qActions = np.where((QPlayer._qtable[currentState,:] == maxQVal))
            qActions = qActions[0]
            qActions += 1 # Maps the action ind to action
            print_debug("Q Actions:",qActions)
            for ind in range(len(qActions)):
                matElem = np.where(possActions == qActions[ind])
                if(len(matElem[0])>0):
                    actionSpace.append(qActions[ind])
            
        print_debug("Filtered Actions:",actionSpace)

        randSeed = random.randint(0 ,  len(actionSpace) -1 )
        print_debug("Rand Number:",randSeed)

        action = actionSpace[randSeed]

        print_debug("Random Action Chosen",action)
        numBoxesFilled = boardConfiguration.stateTransition(action)

        # Calculate reward Val
        self._playerScore += numBoxesFilled

        winRewardFlag = 0
        if(self._playerScore >= self._numBoxesForWin):
            winRewardFlag = 1
            self._playerWon = True

        rewardVal = 1*numBoxesFilled + 5*winRewardFlag
        
        newState = boardConfiguration._boardState
        maxQValNewState = np.max(QPlayer._qtable[newState , :])
        print_debug("Max val in new State: ",maxQValNewState)

        if(1 == self._mode):
            QPlayer._qtable[currentState , action-1] = QPlayer._qtable[currentState , action-1] + self._learningRate*(rewardVal + self._discountRate*maxQValNewState - QPlayer._qtable[currentState , action-1])
            print_debug("Updated QTable:",currentState, action,QPlayer._qtable[currentState , action-1])


        return numBoxesFilled

    def get_qTable(self):
        return QPlayer._qtable
        
    def set_qTable(self , val):
        QPlayer._qtable = val

    def setMode(self , setVal):
        self._mode = setVal

    qtable = property(fget = get_qTable, fset = set_qTable)
    mode = property(fset = setMode)

#%%
class QFunctionPlayer(Player):
    def __init__(self , boardConfiguration, learnRate, epsilonVal, discRate, qEstimator):
        Player.__init__(self , boardConfiguration)
        self._learningRate = learnRate
        self._discountRate = discRate
        self._epsilon = epsilonVal
        # Mode 1 refers to training and mode 2 refers to test
        self._mode = 1
        # Creating the Q table
        QFunctionPlayer._estimator = qEstimator

#        self.initializeQtable()

    def chooseAction(self, state, possActions):
        
        epsilonRandSeed = random.uniform(0,1)
        print_debug("Epsilon Rand Seed:",epsilonRandSeed)        
        
        print_debug("Possible Actions:",possActions)
        
        actionSpace = []
        if((epsilonRandSeed < self._epsilon) and (self._mode == 1)):
            actionSpace = possActions          
        else:
            qPossActions = QFunctionPlayer._estimator.predict([state], action=None)
            print_debug("Q possible Actions:",qPossActions)
            maxQval = np.max(qPossActions)
            while(-1 != maxQval):
                qActions = np.where(qPossActions == maxQval)[0]
                qActions = qActions+ 1
                print_debug("Actions with max val:",qActions)
                print_debug("Q Actions:",qActions)
                actionSpace = []
                for ind in range(len(qActions)):
                    matElem = np.where(possActions == qActions[ind])
                    if(len(matElem[0])>0):
                        actionSpace.append(qActions[ind])
                        
                if(len(actionSpace) > 0):
                    break;
                else:
                    qPossActions[qActions-1] = -1
                
                maxQval = np.max(qPossActions)
            
            if(len(actionSpace) == 0):
                actionSpace = possActions
                    
                    
        print_debug("Filtered Action Space:",actionSpace)
        actionRandSeed = random.randint(0,len(actionSpace)-1)
        chosenAction = actionSpace[actionRandSeed]
        print_debug("Chosen Action:",chosenAction)
        
        return chosenAction
            
            
    def drawLine(self, boardConfiguration):
        currentState = boardConfiguration._boardState
        
        possActions = boardConfiguration._possibleActions[currentState][1]
        
        nextAction = self.chooseAction(currentState, possActions)
        
        

#        maxQVal = np.max(QPlayer._qtable[currentState,:])
#        print_debug("Max QVal:",maxQVal)
#
#        exploreRandSeed = random.uniform(0,1)
#        
#        if((exploreRandSeed < self._epsilon) and (self._mode == 1)):
#            matElems = np.where((QPlayer._qtable[currentState,:] >= 0))
#        else:
#            matElems = np.where((QPlayer._qtable[currentState,:] == maxQVal))
#            
#        row = matElems[0]
#        print_debug("Possible Actions:",row)
#
#        randSeed = random.randint(0 ,  row.shape[0] -1 )
#        print_debug("Rand Number:",randSeed)
#
#        action = row[randSeed]

        print_debug("Random Action Chosen",nextAction)
        
        numBoxesFilled = boardConfiguration.stateTransition(nextAction)

        # Calculate reward Val
        self._playerScore += numBoxesFilled

        winRewardFlag = 0
        if(self._playerScore >= self._numBoxesForWin):
            winRewardFlag = 1
            self._playerWon = True

        rewardVal = 1*numBoxesFilled + 5*winRewardFlag
        
        newState = boardConfiguration._boardState
        maxQValNewState = np.max(QFunctionPlayer._estimator.predict([newState]))
        print_debug("Max val in new State: ",maxQValNewState)
        
        td_target = rewardVal + self._discountRate*maxQValNewState
        print_debug("TD Target for update:",td_target)

        if(1 == self._mode):
            QFunctionPlayer._estimator.update([currentState] , (nextAction-1), td_target)
            print_debug("Updated QTable:",currentState, nextAction,QFunctionPlayer._estimator.predict([currentState], (nextAction-1)))


        return numBoxesFilled
    
