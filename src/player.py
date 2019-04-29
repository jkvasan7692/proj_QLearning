#%%
import numpy as np
import argparse
import os, sys
import random

DEBUG = False

#%% - Common Functions written here
def print_debug(*objects):
    if DEBUG == True:
        print(*objects)
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

        self._possibleActions = list()
        for ind1 in range(self._numStates):
            print_debug(ind1)
            data = 2**numLines- 1
            data ^= ind1
            possActions = list()
            actionNum = 1
            while(data != 0):
                if((data & 1 == 1)):
                    possActions.append(actionNum)
                actionNum += 1
                data = data >> 1
            self._possibleActions.append((ind1, possActions))
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

#%%
class Player:
    def __init__(self , boardConfiguration):
        self._playerScore = 0
        self._boardConfiguration = boardConfiguration
        self._playerWon = False
        self._numBoxesForWin = np.ceil(((self._boardConfiguration._boardType**2)/2)+0.1)

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
    def __init__(self, boardConfiguration):
        Player.__init__(self,boardConfiguration)

    def drawLine(self):
        currentState = self._boardConfiguration._boardState

        possibleActionsCurrentState = self._boardConfiguration._possibleActions[currentState][1]
        print_debug("Possible Actions Current",possibleActionsCurrentState)
        randSeed = random.randint(0 ,  len(possibleActionsCurrentState)-1)
        print_debug("Rand Number:",randSeed)

        action = possibleActionsCurrentState[randSeed]
        print_debug("Random Action Chosen",action)
        numBoxesFilled = self._boardConfiguration.stateTransition(action)

        self._playerScore += numBoxesFilled

        if(self._playerScore >= self._numBoxesForWin):
            self._playerWon = True

        return numBoxesFilled


#%%
class QPlayer(Player):
    def __init__(self , boardConfiguration, learnRate, epsilonVal, discRate, qTable):
        Player.__init__(self , boardConfiguration)
        self._learningRate = learnRate
        self._discountRate = discRate
        self._epsilon = epsilonVal
        # Mode 1 refers to training and mode 2 refers to test
        self._mode = 1
        # Creating the Q table
#        self._qtable = qTable
        (numActions, numStates) = self._boardConfiguration.getStateActionCount()
        QPlayer._qtable = -float('Inf') * np.ones((numStates , numActions),'uint')

        self.initializeQtable()

    def initializeQtable(self):
        possActions = self._boardConfiguration.getPossibleActions()
#        print_debug("PossibleActions",possActions)

        for ind in range(len(possActions)):
            validActionForState = np.array(possActions[ind][1])-1
            for ind2 in range(validActionForState.shape[0]):
                QPlayer._qtable[ind,validActionForState[ind2]] = 0
        QPlayer._qtable[ind, :] = 0
        print_debug("Initialized QTable:",QPlayer._qtable)

    def drawLine(self):
        currentState = self._boardConfiguration._boardState

        maxQVal = np.max(QPlayer._qtable[currentState,:])
        print_debug("Max QVal:",maxQVal)

        exploreRandSeed = random.uniform(0,1)
        
        if((exploreRandSeed < self._epsilon) and (self._mode == 1)):
            matElems = np.where((QPlayer._qtable[currentState,:] >= 0))
        else:
            matElems = np.where((QPlayer._qtable[currentState,:] == maxQVal))
            
        row = matElems[0]
        print_debug("Possible Actions:",row)

        randSeed = random.randint(0 ,  row.shape[0] -1 )
        print_debug("Rand Number:",randSeed)

        action = row[randSeed]

        print_debug("Random Action Chosen",action)
        numBoxesFilled = self._boardConfiguration.stateTransition(action+1)

        # Calculate reward Val
        self._playerScore += numBoxesFilled

        winRewardFlag = 0
        if(self._playerScore >= self._numBoxesForWin):
            winRewardFlag = 1
            self._playerWon = True

        rewardVal = 1*numBoxesFilled + 5*winRewardFlag
        
        newState = self._boardConfiguration._boardState
        maxQValNewState = np.max(QPlayer._qtable[newState , :])
        print_debug("Max val in new State: ",maxQValNewState)

        if(1 == self._mode):
            QPlayer._qtable[currentState , action] = QPlayer._qtable[currentState , action] + self._learningRate*(rewardVal + self._discountRate*maxQValNewState - QPlayer._qtable[currentState , action])
            print_debug("Updated QTable:",currentState, action,QPlayer._qtable[currentState , action])


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

boardConfiguration = BoardConfiguration(boardType=2)
(numActions , numStates) = boardConfiguration.getStateActionCount()

Qtable = -float('Inf') * np.ones((numStates , numActions),'uint')


player1 = RandomPlayer(boardConfiguration)
player2 = QPlayer(boardConfiguration , 0.5 , 0.5, 0.2, Qtable)
player3 = QPlayer(boardConfiguration , 0.5 , 0.5, 0.2, Qtable)
iterationCount = 1
numOfGames = 10000
playersTurn = 0

player2.mode = 1
player3.mode = 1
for ind in range(numOfGames):
    boardConfiguration.boardState = 0
    iterationCount = 1
    player2.score = 0
    player3.score = 0
    player2.wonFlag = False
    player3.wonFlag = False
    while((False == boardConfiguration.wholeBoardFilled()) and (False == player2.wonFlag) and (False == player3.wonFlag)):
        print_debug("Turn Count",iterationCount)
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            numOfBoxesAttained = player2.drawLine()
        else:
            print_debug("Player 2 Turn")
            numOfBoxesAttained = player3.drawLine()
        iterationCount += 1
        if(0 == numOfBoxesAttained):
            playersTurn +=1
#        playersTurn += 1
        playersTurn %= 2
    playerQTable = player2.qtable
    print_debug("QTable positive values at:",np.where(playerQTable > 0))

#%% - Testing the trained player

DEBUG = False
player2.mode = 2
testNumOfGames = 100
player1Wins = 0
player2Wins = 0
playersTurn = 0

print("Playing test games")

for ind in range(testNumOfGames):
    print("Test Game Number:",ind)
    boardConfiguration.boardState = 0
    player1.score = 0
    player2.score = 0

    player1.wonFlag = False
    player2.wonFlag = False
    
    print("BoardState: ","{0:012b}".format(boardConfiguration.boardState))

    while((False == boardConfiguration.wholeBoardFilled()) and (False == player1.wonFlag) and (False == player2.wonFlag)):
        if(1 == playersTurn):
            print("Player 1 turn")
            testNumOfBoxesAttaines = player1.drawLine()
        else:
            print("Player 2 turn")
            testNumOfBoxesAttaines = player2.drawLine()
        if(0 == testNumOfBoxesAttaines):
            playersTurn += 1
        playersTurn %= 2
        print("BoardState: ","{0:012b}".format(boardConfiguration.boardState))

    if(True == player1.wonFlag):
        player1Wins += 1
        print("Player 2 Lost")
    elif(True == player2.wonFlag):
        player2Wins += 1
        print("Player 2 won")

print("Num of Wins in 100 games for Player1:", player1Wins)
print("Num of wins in 100 games for player2:", player2Wins)

#%% - Save onto CSV file
np.savetxt("qtable.csv",playerQTable,delimiter=',')
