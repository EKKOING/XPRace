# XPRace

An ongoing research project into applying the NEAT algorithm to a new framework for XPilot to allow timed races.

## Project Proposal
I am attempting to design a framework that will allow XPilot to serve as a sandbox for training and evaluating agents capable of time trial style racing around various circuits. Once the framework is complete I will use the NEAT algorithm to create capable, human-like agents, and robust agents that not only learn to complete tracks, but over time become faster too!  

## About Xpilot
![XPilot Screenshot](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Xpilotscreen.jpg/1024px-Xpilotscreen.jpg)  
*A Screenshot of an XPilot Game*  
>XPilot is an open-source 2-dimensional space combat simulator which is playable over the internet. Multiple players can connect to a central XPilot server and compete in many varieties of game play, such as free-for-all combat, capture-the-flag, or team combat. Each player controls a space-ship that can turn, thrust, and shoot. There is often a variety of weapons and ship upgrades available on the particular map in which they play. The game uses synchronized client/server networking to allow for solid network play.  

\- [The XPilot-AI Homepage](https://xpilot-ai.com)  
More information on the history of the game's development can be found [here](https://en.wikipedia.org/wiki/XPilot).  

## About Genetic Algorithms
Genetic Algorithms are a form of computational intelligence that through mimicing the natural world's evolution can perform complex optimizations. This optimization is not performed through error correction, but rather through semi-random adjustments. 
The basic structure of a GA is as follows:
Create a population of individuals.
Evaluate the fitness of each individual in the population.
Select a subset of the population to undergo genetic operations.
Perform genetic operations on the selected subset.
Repeat steps 2 and 3 until a satisfactory solution is found.

## About NEAT
NEAT is a genetic algorithm that is specifically designed to solve the problem of evolving neural networks. It is a form of artificial intelligence that is used to solve problems that are difficult to solve using conventional genetic algorithms or backpropagation.
NEAT modifies the general GA structure by introducing a number of innovations to the GA. These innovations include:
 - The ability to modify the structure of the NN as well as the weights and biases of the NN.
 - Adding speciation, which allows individuals to only compete amongst those similar to them until the strategy can develop.

The former is the greatest strength of NEAT as it allows for the network to increase in complexity to meet whatever challenge is presented to the algorithm. The latter is a great help to ensuring that the algorithm does not suffer from the drawbacks of a standard GA being applied to a NN.
