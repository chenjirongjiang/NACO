# Natural Computing assignment 21/22

This repository contains all files required to complete the assignment for the NACO course 2021. Briefly, this project tests the performance of various mutation, recombination and selection operators in a Genetic Algorithm on various Cellular Automata inverse problems, see the report for more detail. 

The repository contains the following files:

- Code/: Contains all code necessary to run the Genetic Algorithm (GA), Cellular Automata (CA) and the application of the GA to solve the inverse problem.
- Group_3_NaCo_project.pdf: A report on the performance of various methods for the mutation, recombination and selection of the GA on the different parameters in the CA inverse problem.
- requirements.txt: A list of packages of the python environment used to run this project.

The Code folder contains the following files:
- algorithm.py: Defines the Algirthm superclass for the GA.
- ca_input.csv: Contains the problem parameters that are used for the performance tests.
- implementation.py: Defines the RandomSearch and GeneticAlgorithm classes as well as the different methods that are to be tested.
- objective_function.py: Defines the CellularAutomata class that is used to simulate the problem set as well as the code that runs the experiments.
- test_algorithm.py: Contains code that connects ioh problems with the GA.


## Installing requirements
You are encouraged to use a virtual environment. Install the required dependecies with the following command:
```bash
$ pip install -r requirements.txt
```
