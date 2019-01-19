# Neural Network Analyser

Neural Network analyzer produces detailed information about the estimated execution time, memory required, MACs and other metrics for a given networks. 

## These parameters will serve as inputs to your performance prediction tool
-External – internal memory bandwidth in bytes / second (10 GB / second) Internal memory size in bytes (16 MB) 

-Matrix multiplication primitive size M, N, K in BLAS notation (32, 32, 32) Number of matrix multiplication primitives operating in     parallel (1) Number of matrix multiplication primitive completions per second (1e9 / 32) 

-Vector primitive size N x 1 (32) Number of vector primitives operating in parallel (1) Number of vector primitive completions per        second (1e9) 


## Estimated parameters
-Input feature map up sampling ratio 
-Number of input feature maps 
-Number of input feature map rows 
-Number of input feature map cols 
-Number of input feature map bytes 

-Output feature map location (external or internal memory) 
-Output feature map down sampling ratio 
-Number of output feature maps 
-Number of output feature map rows 
-Number of output feature map cols 
-Number of output feature map bytes 

-Filter coefficient location (external memory) 
-Filter up sampling ratio 
-Filter grouping 
-Number of filter rows 
-Number of filter cols 
-Number of filter bytes 

-Input feature map data movement time 
-Filter coefficient data movement time 
-Output feature map data movement time 
-Total data movement time for the layer 

-Matrix compute time 
-Vector compute time 
-Total compute time for the layer 

-Serial total data movement and compute time for the layer 
-Total data movement time for the layer + total compute time for the layer 
-Parallel total data movement and compute time for the layer 

-Input feature map data movement time 
-Filter coefficient data movement time 
-Output feature map data movement time 
-Total data movement time for the network 

-Matrix compute time 
-Vector compute time 
-Total compute time for the network 

-Sum of serial total data movement and compute time for every layer 
-Sum of parallel total data movement and compute time for every layer


The .json files are generated using jsonGeneration.py from Tensorflow Keras Library.
Change the parameters from the "# Define" section at the beginning of analysis.py

Assumptions:
1. No compute considered for padding layers.
2. Memory location is 1 if the data is in internal memory and 0 if the data is in external memory.
