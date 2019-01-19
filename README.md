# Neural Network Analyser

Neural Network analyzer produces detailed information about the estimated execution time, memory required, MACs and other metrics for a given networks. 

## These parameters will serve as inputs to your performance prediction tool
1. External – internal memory bandwidth in bytes / second (10 GB / second) Internal memory size in bytes (16 MB) 

2. Matrix multiplication primitive size M, N, K in BLAS notation (32, 32, 32) Number of matrix multiplication primitives operating in     parallel (1) Number of matrix multiplication primitive completions per second (1e9 / 32) 

3. Vector primitive size N x 1 (32) Number of vector primitives operating in parallel (1) Number of vector primitive completions per        second (1e9) 


## Estimated parameters
1. Input feature map up sampling ratio
2. Number of input feature maps 
3. Number of input feature map rows 
4. Number of input feature map cols 
5. Number of input feature map bytes 

6. Output feature map location (external or internal memory) 
7. Output feature map down sampling ratio 
8. Number of output feature maps 
9. Number of output feature map rows 
10. Number of output feature map cols 
11. Number of output feature map bytes 

12. Filter coefficient location (external memory) 
13. Filter up sampling ratio 
14. Filter grouping 
15. Number of filter rows 
16. Number of filter cols 
17. Number of filter bytes 

18. Input feature map data movement time 
19. Filter coefficient data movement time 
20. Output feature map data movement time 
21. Total data movement time for the layer 

22. Matrix compute time 
23. Vector compute time 
24. Total compute time for the layer 

25. Serial total data movement and compute time for the layer 
26. Total data movement time for the layer + total compute time for the layer 
27. Parallel total data movement and compute time for the layer 

28. Input feature map data movement time 
29. Filter coefficient data movement time 
30. Output feature map data movement time 
31. Total data movement time for the network 

32. Matrix compute time 
33. Vector compute time 
34. Total compute time for the network 

35. Sum of serial total data movement and compute time for every layer 
36. Sum of parallel total data movement and compute time for every layer


The .json files are generated using jsonGeneration.py from Tensorflow Keras Library.
Change the parameters from the "# Define" section at the beginning of analysis.py

Assumptions:
1. No compute considered for padding layers.
2. Memory location is 1 if the data is in internal memory and 0 if the data is in external memory.
