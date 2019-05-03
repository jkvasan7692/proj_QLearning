# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:07:40 2019

@author: kirthi
"""
#%%
import numpy as np
import argparse
import os, sys
import random
from common import *
from boardConfiguration import *
from player import *

common.DEBUG = False

#%% Main program starts here
#%% - Initialize board and random player
board3Configuration = BoardConfiguration(boardType=3)

#%%
player1_b3_rand = RandomPlayer(3)
#%%
trainNumOfGames = 100

testNumOfGames = 1000
playerTurn = 0


#%% Q Table board 3 training 
print("Training for Board 3 using Q table and seed learning")

(numActions , numStates) = board3Configuration.getStateActionCount()

Qtable = np.zeros((numStates , numActions),'uint')

#%% Seeding using the QTable board2
mask1 = int('11',2)
mask2 = int('1100',2)
mask3 = int('110000',2)
mask4 = int('11000000',2)
mask5 = int('1100000000',2)
mask6 = int('110000000000',2)
QTable2 = np.loadtxt("qtableBoard2.csv",delimiter=",")
numLinesPerRow = 6

for ind1 in range(QTable2.shape[0]):
    print_debug(bin(ind1))
    stateInd = (ind1 & mask1) | ((ind1 & mask2) << 1) | ((ind1 & mask3) << 2) | ((ind1 & mask4) << 3) | ((ind1 & mask5) << 4) | ((ind1 & mask6) << 5)
    print_debug(bin(stateInd))
    for ind2 in range(QTable2.shape[1]):
        if(ind2 < numLinesPerRow):
            actionInd = ind2 + np.floor(ind2/2)
        else:
            actionInd = ind2 + numLinesPerRow + np.floor((ind2 - numLinesPerRow)/3)
        Qtable[int(stateInd) , int(actionInd)] = QTable2[ind1 , ind2]
        print_debug(actionInd)

#%%

player2_b3_table = QPlayer(3 , 0.5 , 0.5, 0.2, Qtable)
player3_b3_table = QPlayer(3 , 0.5 , 0.5, 0.2, Qtable)
iterationCount = 1
playersTurn = 0

player2_b3_table.mode = 1
player3_b3_table.mode = 1
for ind in range(trainNumOfGames):
    board3Configuration.boardState = 0
    iterationCount = 1
    player2_b3_table.score = 0
    player3_b3_table.score = 0
    player2_b3_table.wonFlag = False
    player3_b3_table.wonFlag = False
    while((False == board3Configuration.wholeBoardFilled()) and (False == player2_b3_table.wonFlag) and (False == player3_b3_table.wonFlag)):
        print_debug("Turn Count",iterationCount)
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            numOfBoxesAttained = player2_b3_table.drawLine(board3Configuration)
        else:
            print_debug("Player 2 Turn")
            numOfBoxesAttained = player3_b3_table.drawLine(board3Configuration)
        iterationCount += 1
        if(0 == numOfBoxesAttained):
            playersTurn +=1
#        playersTurn += 1
        playersTurn %= 2
    playerQTable = player2_b3_table.qtable
    print_debug("QTable positive values at:",np.where(playerQTable > 0))

#%% - Testing the trained player

DEBUG = False
player2_b3_table.mode = 2
player1_b3_randWins = 0
player2_b3_tableWins = 0
playersTurn = 0

print("Testing for Board 3 using Q Table")

for ind in range(testNumOfGames):
    print_debug("Test Game Number:",ind)
    board3Configuration.boardState = 0
    player1_b3_rand.score = 0
    player2_b3_table.score = 0

    player1_b3_rand.wonFlag = False
    player2_b3_table.wonFlag = False

    print_debug("BoardState: ","{0:012b}".format(board3Configuration.boardState))

    while((False == board3Configuration.wholeBoardFilled()) and (False == player1_b3_rand.wonFlag) and (False == player2_b3_table.wonFlag)):
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            testNumOfBoxesAttaines = player1_b3_rand.drawLine(board3Configuration)
        else:
            print_debug("Player 2 turn")
            testNumOfBoxesAttaines = player2_b3_table.drawLine(board3Configuration)
        if(0 == testNumOfBoxesAttaines):
            playersTurn += 1
        playersTurn %= 2
        print_debug("BoardState: ","{0:012b}".format(board3Configuration.boardState))

    if(True == player1_b3_rand.wonFlag):
        player1_b3_randWins += 1
        print_debug("Player 2 Lost")
    elif(True == player2_b3_table.wonFlag):
        player2_b3_tableWins += 1
        print_debug("Player 2 won")

print("Num of Wins in % for Random Player:", (player1_b3_randWins/testNumOfGames)*100)
print("Num of wins in % for Q Table Player:", (player2_b3_tableWins/testNumOfGames)*100)
#%% - Save onto CSV file  QTable
#np.savetxt("qtableBoard3.csv",playerQTable,delimiter=',')

#%% Q Function Approximator board 3 training 
print("Training for Board 3 using QFunction Approximator")
(numActions , numStates) = board2Configuration.getStateActionCount()
QEstimator = FunctionApproximator(numActions, numStates)

player2_b3_function = QPlayer(board3Configuration , 0.5 , 0.5, 0.2, QEstimator)
player3_b3_function = QPlayer(board3Configuration , 0.5 , 0.5, 0.2, QEstimator)
iterationCount = 1
playersTurn = 0

player2_b3_function.mode = 1
player3_b3_function.mode = 1
for ind in range(trainNumOfGames):
    board3Configuration.boardState = 0
    iterationCount = 1
    player2_b3_function.score = 0
    player3_b3_function.score = 0
    player2_b3_function.wonFlag = False
    player3_b3_function.wonFlag = False

    while((False == board3Configuration.wholeBoardFilled()) and (False == player2_b3_function.wonFlag) and (False == player3_b3_function.wonFlag)):
        print_debug("Turn Count",iterationCount)
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            numOfBoxesAttained = player2_b3_function.drawLine()
        else:
            print_debug("Player 2 Turn")
            numOfBoxesAttained = player3_b3_function.drawLine()
        iterationCount += 1
        if(0 == numOfBoxesAttained):
            playersTurn +=1
#        playersTurn += 1
        playersTurn %= 2


#%% Q Function approximator board 3 testing
DEBUG = False
player2_b3_function.mode = 2
player1_b3_randWins = 0
player2_b3_functionWins = 0
playersTurn = 0

print("Testing for Board 3 using Qfunction approximator")

for ind in range(testNumOfGames):
    print_debug("Test Game Number:",ind)
    board2Configuration.boardState = 0
    player1_b3_rand.score = 0
    player2_b3_function.score = 0

    player1_b3_rand.wonFlag = False
    player2_b3_function.wonFlag = False

    print_debug("BoardState: ","{0:012b}".format(board3Configuration.boardState))

    while((False == board3Configuration.wholeBoardFilled()) and (False == player1_b3_rand.wonFlag) and (False == player2_b3_function.wonFlag)):
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            testNumOfBoxesAttaines = player1_b3_rand.drawLine()
        else:
            print_debug("Player 2 turn")
            testNumOfBoxesAttaines = player2_b3_function.drawLine()
        if(0 == testNumOfBoxesAttaines):
            playersTurn += 1
        playersTurn %= 2
        print_debug("BoardState: ","{0:012b}".format(board2Configuration.boardState))

    if(True == player1_b2_rand.wonFlag):
        player1_b3_randWins += 1
        print_debug("Player 2 Lost")
    elif(True == player2_b3_function.wonFlag):
        player2_b3_functionWins += 1
        print_debug("Player 2 won")

print("Num of Wins in % for Random player:", (player1_b3_randWins/testNumOfGames)*100)
print("Num of wins in % for Q Player with function approximator:", (player2_b3_functionWins/testNumOfGames)*100)

