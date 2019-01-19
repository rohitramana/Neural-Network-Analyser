# Neural Network Analyser

Neural Network analyzer produces detailed information about the estimated execution time, memory required, MACs and other metrics for a given networks. 

The .json files are generated using jsonGeneration.py from Tensorflow Keras Library.
Change the parameters from the "# Define" section at the beginning of analysis.py

Assumptions:
1. No compute considered for padding layers.
2. Memory location is 1 if the data is in internal memory and 0 if the data is in external memory.
