# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:23:28 2019

@author: kirthi
"""
#%%
import numpy as np
import argparse
import os, sys
import random
from common import *
from FunctionApproximator import *
from boardConfiguration import *
from player import *


common.DEBUG = False
#%% Main program starts here
#%% - Initialize board and random player
board2Configuration = BoardConfiguration(boardType=2)

player1_b2_rand = RandomPlayer(2)
trainNumOfGames = 10000

testNumOfGames = 1000
playerTurn = 0

#%% - Board Type 2 using Q table
print("Training for Board 2 using Q Table")

(numActions , numStates) = board2Configuration.getStateActionCount()

Qtable = np.zeros((numStates , numActions),'uint')


player2_b2_table = QPlayer(2 , 0.5 , 0.2, 0.5, Qtable)
player3_b2_table = QPlayer(2 , 0.5 , 0.2, 0.5, Qtable)
iterationCount = 1
playersTurn = 0

player2_b2_table.mode = 1
player3_b2_table.mode = 1
for ind in range(trainNumOfGames):
    board2Configuration.boardState = 0
    iterationCount = 1
    player2_b2_table.score = 0
    player3_b2_table.score = 0
    player2_b2_table.wonFlag = False
    player3_b2_table.wonFlag = False
    while((False == board2Configuration.wholeBoardFilled()) and (False == player2_b2_table.wonFlag) and (False == player3_b2_table.wonFlag)):
        print_debug("Turn Count",iterationCount)
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            numOfBoxesAttained = player2_b2_table.drawLine(board2Configuration)
        else:
            print_debug("Player 2 Turn")
            numOfBoxesAttained = player3_b2_table.drawLine(board2Configuration)
        iterationCount += 1
        if(0 == numOfBoxesAttained):
            playersTurn +=1
#        playersTurn += 1
        playersTurn %= 2
    playerQTable = player2_b2_table.qtable
    print_debug("QTable positive values at:",np.where(playerQTable > 0))

#%% - Testing the trained player

common.DEBUG = False
player2_b2_table.mode = 2
player1_b2_randWins = 0
player2_b2_tableWins = 0
playersTurn = 0

print("Testing for Board 2 using Q Table")

for ind in range(testNumOfGames):
    print_debug("Test Game Number:",ind)
    board2Configuration.boardState = 0
    player1_b2_rand.score = 0
    player2_b2_table.score = 0

    player1_b2_rand.wonFlag = False
    player2_b2_table.wonFlag = False

    print_debug("BoardState: ","{0:012b}".format(board2Configuration.boardState))

    while((False == board2Configuration.wholeBoardFilled()) and (False == player1_b2_rand.wonFlag) and (False == player2_b2_table.wonFlag)):
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            testNumOfBoxesAttaines = player1_b2_rand.drawLine(board2Configuration)
        else:
            print_debug("Player 2 turn")
            testNumOfBoxesAttaines = player2_b2_table.drawLine(board2Configuration)
        if(0 == testNumOfBoxesAttaines):
            playersTurn += 1
        playersTurn %= 2
        print_debug("BoardState: ","{0:012b}".format(board2Configuration.boardState))

    if(True == player1_b2_rand.wonFlag):
        player1_b2_randWins += 1
        print_debug("Player 2 Lost")
    elif(True == player2_b2_table.wonFlag):
        player2_b2_tableWins += 1
        print_debug("Player 2 won")

print("Num of Wins in % for Random Player:", (player1_b2_randWins/testNumOfGames)*100)
print("Num of wins in % for Q Table Player:", (player2_b2_tableWins/testNumOfGames)*100)

#%% - Save onto CSV file  QTable
np.savetxt("qtableBoard2.csv",playerQTable,delimiter=',')

#%% - QFunction Approximator Board 2 initialization
print("Training for Board 2 using QFunction Approximator")
(numActions , numStates) = board2Configuration.getStateActionCount()
QEstimator = FunctionApproximator(numActions, numStates)

player2_b2_function = QFunctionPlayer(2 , 0.5 , 0.2, 0.5, QEstimator)
player3_b2_function = QFunctionPlayer(2 , 0.5 , 0.2, 0.5, QEstimator)
iterationCount = 1
playersTurn = 0

player2_b2_function.mode = 1
player3_b2_function.mode = 1
for ind in range(trainNumOfGames):
    board2Configuration.boardState = 0
    iterationCount = 1
    player2_b2_function.score = 0
    player3_b2_function.score = 0
    player2_b2_function.wonFlag = False
    player3_b2_function.wonFlag = False

    while((False == board2Configuration.wholeBoardFilled()) and (False == player2_b2_function.wonFlag) and (False == player3_b2_function.wonFlag)):
        print_debug("Turn Count",iterationCount)
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            numOfBoxesAttained = player2_b2_function.drawLine(board2Configuration)
        else:
            print_debug("Player 2 Turn")
            numOfBoxesAttained = player3_b2_function.drawLine(board2Configuration)
        iterationCount += 1
        if(0 == numOfBoxesAttained):
            playersTurn +=1
#        playersTurn += 1
        playersTurn %= 2

#%% QFunction Approximator Board 2 testing
common.DEBUG = False
player2_b2_function.mode = 2
player1_b2_randWins = 0
player2_b2_functionWins = 0
playersTurn = 0

print("Testing for Board 2 using Qfunction approximator")

for ind in range(testNumOfGames):
    print_debug("Test Game Number:",ind)
    board2Configuration.boardState = 0
    player1_b2_rand.score = 0
    player2_b2_function.score = 0

    player1_b2_rand.wonFlag = False
    player2_b2_function.wonFlag = False

    print_debug("BoardState: ","{0:012b}".format(board2Configuration.boardState))

    while((False == board2Configuration.wholeBoardFilled()) and (False == player1_b2_rand.wonFlag) and (False == player2_b2_function.wonFlag)):
        if(1 == playersTurn):
            print_debug("Player 1 turn")
            testNumOfBoxesAttaines = player1_b2_rand.drawLine(board2Configuration)
        else:
            print_debug("Player 2 turn")
            testNumOfBoxesAttaines = player2_b2_function.drawLine(board2Configuration)
        if(0 == testNumOfBoxesAttaines):
            playersTurn += 1
        playersTurn %= 2
        print_debug("BoardState: ","{0:012b}".format(board2Configuration.boardState))

    if(True == player1_b2_rand.wonFlag):
        player1_b2_randWins += 1
        print_debug("Player 2 Lost")
    elif(True == player2_b2_function.wonFlag):
        player2_b2_functionWins += 1
        print_debug("Player 2 won")

print("Num of Wins in % for Random player:", (player1_b2_randWins/testNumOfGames)*100)
print("Num of wins in % for Q Player with function approximator:", (player2_b2_functionWins/testNumOfGames)*100)