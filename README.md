# Dots and Boxes using Q Learning

This project describes the dots and boxes game for a 2x2 and 3x3 board played using Q Learning. In this project the player is trained in Q Learning with both the cases of Q Table and function approximation.

## Getting Started

### Prerequisites

The code is implemented in Python and has the following Dependency :
1. Python 3 or Python 2.7
2. Scikit library for python

### Installing

Create a folder in which you would like to clone the project repository.

Execute the following command:
git clone https://github.com/jkvasan7692/proj_QLearning.git

### Directory and Files

The project directory structure is as follows:
1. src - Contains the source code Files
  * Board2Program.py - Main code for executing the 2x2 board using Q Table and Function Approximator
  * Board3Program.py - Main code for executing the 3x3 board using Q Table and Function Approximator
  * boardConfiguration.py - Class for the board states and possible boardConfiguration and board transition
  * player.py - Player class which is inherited to produce RandomPlayer, QPlayer and QFunctionPlayer class. The class contains the source code to draw the line on the board and maintain the scores of the player
  * FunctionApproximator.py - Radial Basis Function Approximator class to implement a function approximation of Q Learning values
2. Report.pdf - Description of the boards and the results
3. Results - Contains the output logs of the execution for 2x2 and 3x3 board

## Running the tests

### Execution for 2x2 board
1. Open a terminal and goto src folder.
2. Execute the following command:
python3 Board2Program.py
3. To change the number of training or test games, the values of the variable trainNumOfGames, testNumOfGames has to be changed in Board2Program.py before execution

The program executes showing the win percentage Results of the RandomPlayer and Q Player

### Execution for 3x3 board
1. Open a terminal and goto src folder.
2. Execute the following command:
python3 Board3Program.py
3. To change the number of training or test games, the values of the variable trainNumOfGames, testNumOfGames has to be changed in Board3Program.py before execution

The program executes showing the win percentage Results

## Authors

**Janakiraman Kirthivasan** - *Initial work* - [jkvasan7692](https://github.com/jkvasan7692)

## Acknowledgments

* The following link was referred to get the idea of function approximation: https://github.com/dennybritz/reinforcement-learning/tree/master/FA
