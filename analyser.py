import json
import math
import pandas as pd

# # # # #
# DEFINES
inputJson = "inceptionv3.json"
memoryBandwidth = 10000000000    # 10GB
internalMemorySize = 16000000	 # 16MB	
 
matrixM = 32
matrixN = 32
matrixK = 32

matrixPrimitivesInParallel = 1
matrixCompletionsPerSec = 1e9 / 32

vectorN = 32

vectorPrimitivesInParallel = 1
vectorPrimitivesPerSec = 1e9


# # # # #



with open(inputJson, "r") as read_file:
	data = json.load(read_file)

layer_list = data['config']['layers']

output_dims_dict = {}

complete_total_info_array = []

total_info_array = []

layer_info_array = []

freeInternalMemory = internalMemorySize

class OutputDims:

	def __init__(self, rows, columns, channels):
		self.channels = channels
		self.rows = rows
		self.columns = columns


totalmacc = 0
totalparams = 0
totalcomp = 0

for layer in layer_list:
	if len(layer['inbound_nodes']) > 0:
		
		if layer['class_name'] == 'Concatenate':
			inputMemory = 0
			sum_channels = 0


			for prev_layer in layer['inbound_nodes'][0]:
				prev_layer_output = output_dims_dict[prev_layer[0]]
				sum_channels = sum_channels + prev_layer_output.channels
				inputFeatureMaps = (prev_layer_output.channels, prev_layer_output.rows, prev_layer_output.columns)
				inputMemory = inputMemory + inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMemory = inputMemory - inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
				freeInternalMemory = internalMemorySize - inputMemory
			else:
				freeInternalMemory = internalMemorySize

			inputMovementTime = inputMemory / memoryBandwidth

			inputMemory = inputMemory + inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

		
			outputFeatureMaps = (sum_channels, inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory/ memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, inputMovementTime + outputMovementTime, 0, 0, 0, totalMovementTime, max(totalMovementTime, 0)]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'AveragePooling2D':
			poolSize = layer['config']['pool_size']
			stride = layer['config']['strides']

			if layer['config']['padding'] == 'same':
				padding = (filterMaps[2]-1, filterMaps[3]-1)
			elif layer['config']['padding'] == 'valid':
				padding = (0, 0)

			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 0:
				inputMovementTime = inputMemory / memoryBandwidth
			else:
				inputMovementTime = 0
			freeInternalMemory = internalMemorySize - inputMemory
			

			computeTime = inputMemory / (vectorN * vectorPrimitivesPerSec * vectorPrimitivesInParallel)
			totalComputeTime = computeTime


			outputFeatureMaps = (filterMaps[0], math.ceil((inputFeatureMaps[1] + padding[1] - (poolSize[1] - 1)) / stride[0]), math.ceil((inputFeatureMaps[2] + padding[0] - (poolSize[0] - 1))/stride[1]))
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory/ memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, stride[0], outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'Add':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]
			inputMemory = 0
			sum_channels = 0

			for prev_layer in layer['inbound_nodes'][0]:
				prev_layer_output = output_dims_dict[prev_layer[0]]
				sum_channels = sum_channels + prev_layer_output.channels
				inputFeatureMaps = (prev_layer_output.channels, prev_layer_output.rows, prev_layer_output.columns)
				inputMemory = inputMemory + inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMemory = inputMemory - inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			inputMovementTime = inputMemory / memoryBandwidth
			inputMemory = inputMemory + inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			freeInternalMemory = internalMemorySize - inputMemory

			computeTime = inputMemory / (vectorN * vectorPrimitivesPerSec * vectorPrimitivesInParallel)
			totalComputeTime = computeTime


			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory/ memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			
			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'Activation':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory/ memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = inputMemory / (vectorN * vectorPrimitivesPerSec * vectorPrimitivesInParallel)
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation
			

		elif layer['class_name'] == 'Dropout':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory
			

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory/ memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = 0
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]
		
			
			inputLocation = outputLocation

		elif layer['class_name'] == 'Reshape':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = 0
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]
			
			
			inputLocation = outputLocation

		elif layer['class_name'] == 'GlobalAveragePooling2D':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory

			outputFeatureMaps = (inputFeatureMaps[0], 1, 1)
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = inputMemory / (vectorN * vectorPrimitivesPerSec * vectorPrimitivesInParallel)
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			
			layer_info_array = [layer['name'], inputLocation, inputFeatureMaps[1], inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'BatchNormalization':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1],inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = inputMemory / (vectorN * vectorPrimitivesPerSec * vectorPrimitivesInParallel)
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]
			
			
			inputLocation = outputLocation

		elif layer['class_name'] == 'ReLU':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]

			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth

			freeInternalMemory = internalMemorySize - inputMemory

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			computeTime = 0
			totalComputeTime = computeTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'Conv2D':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory


			filterMaps = (layer['config']['filters'], inputFeatureMaps[0], layer['config']['kernel_size'][0], layer['config']['kernel_size'][1])
			filterLocation = 0
			filterGrouping = filterMaps[0]
			filterMemory = filterMaps[0] * filterMaps[1] * filterMaps[2] * filterMaps[3]
			filterMovementTime = filterMemory / memoryBandwidth

			stride = (layer['config']['strides'][0], layer['config']['strides'][1])

			if layer['config']['padding'] == 'same':
				padding = (filterMaps[2]-1, filterMaps[3]-1)
			elif layer['config']['padding'] == 'valid':
				padding = (0, 0)

			inputToeplitz = (inputFeatureMaps[0] * filterMaps[2] * filterMaps[3], math.ceil((inputFeatureMaps[2] + padding[1] - (filterMaps[3] - 1))/stride[1]) * math.ceil((inputFeatureMaps[1] + padding[0] - (filterMaps[2] - 1)) / stride[0]))
			filterToeplitz = (filterMaps[0], inputFeatureMaps[0] * filterMaps[2] * filterMaps[3])
			outputToeplitz = (filterMaps[0], inputToeplitz[1])

			computeTime = (filterToeplitz[0] * filterToeplitz[1] * inputToeplitz[1]) / (matrixK * matrixN * matrixM * matrixPrimitivesInParallel * matrixCompletionsPerSec)
			totalComputeTime = computeTime

			outputFeatureMaps = (filterMaps[0], math.ceil((inputFeatureMaps[1] + padding[1] - (filterMaps[2] - 1)) / stride[0]), math.ceil((inputFeatureMaps[2] + padding[0] - (filterMaps[3] - 1))/stride[1]))
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime + filterMovementTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, stride[0], outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, filterLocation, 1, filterGrouping, filterMaps[2], filterMaps[3], filterMemory, inputMovementTime, filterMovementTime, outputMovementTime, totalMovementTime, computeTime, 0, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'MaxPooling2D':
			
			poolSize = layer['config']['pool_size']
			stride = layer['config']['strides']

			if layer['config']['padding'] == 'same':
				padding = (filterMaps[2]-1, filterMaps[3]-1)
			elif layer['config']['padding'] == 'valid':
				padding = (0, 0)

			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]
			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory

			computeTime = inputMemory / (vectorN * vectorPrimitivesInParallel * vectorPrimitivesPerSec)
			totalComputeTime = computeTime

			outputFeatureMaps = (filterMaps[0], math.ceil((inputFeatureMaps[1] + padding[1] - (poolSize[1] - 1)) / stride[0]), math.ceil((inputFeatureMaps[2] + padding[0] - (poolSize[0] - 1))/stride[1]))
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime
			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, stride[0], outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] =='Flatten':

			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]
			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory

			computeTime = 0
			totalComputeTime = computeTime

			outputFeatureMaps = (inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2], 1, 1)
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, 0, computeTime, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'Dense':

			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory

			units = layer['config']['units'] 

			weightMatrix = (units, inputFeatureMaps[0])

			comp = 0
			if layer['config']['activation'] == 'softmax':
				comp = (outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]) / (vectorN * vectorPrimitivesInParallel * vectorPrimitivesPerSec)

			computeTime = (weightMatrix[0] * weightMatrix[1] * 1)  / (matrixK * matrixN * matrixM * matrixPrimitivesInParallel * matrixCompletionsPerSec) + comp
			totalComputeTime = computeTime

			outputFeatureMaps = (units, 1, 1)
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime 

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, computeTime, 0, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'ZeroPadding2D':

			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory

			computeTime = 0
			totalComputeTime = computeTime

			padding = layer['config']['padding']

			outputFeatureMaps = (inputFeatureMaps[0], inputFeatureMaps[1] + padding[0][0] + padding[0][1], inputFeatureMaps[2] + padding[1][0] + padding[1][1])
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])
			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, outputMovementTime, totalMovementTime, computeTime, 0, totalComputeTime, seriesTotalTime, parallelTotalTime]

			
			inputLocation = outputLocation

		elif layer['class_name'] == 'DepthwiseConv2D':
			input_class = output_dims_dict[layer['inbound_nodes'][0][0][0]]

			if layer['config']['padding'] == 'same':
				padding = (filterMaps[2]-1, filterMaps[3]-1)
			elif layer['config']['padding'] == 'valid':
				padding = (0, 0)

			stride = (layer['config']['strides'][0], layer['config']['strides'][1])

			inputFeatureMaps = (input_class.channels, input_class.rows, input_class.columns)
			inputMemory = inputFeatureMaps[0] * inputFeatureMaps[1] * inputFeatureMaps[2]
			if inputLocation == 1:
				inputMovementTime = 0
			else:
				inputMovementTime = inputMemory / memoryBandwidth
			freeInternalMemory = internalMemorySize - inputMemory


			filterMaps = (inputFeatureMaps[0], 1, layer['config']['kernel_size'][0], layer['config']['kernel_size'][1])
			filterLocation = 0
			filterGrouping = filterMaps[0]
			filterMemory = filterMaps[0] * filterMaps[1] * filterMaps[2] * filterMaps[3]
			filterMovementTime = filterMemory / memoryBandwidth			

			inputToeplitz = (inputFeatureMaps[0] * filterMaps[2] * filterMaps[3], math.ceil((inputFeatureMaps[2] + padding[1] - (filterMaps[3] - 1))/stride[1]) * math.ceil((inputFeatureMaps[1] + padding[0] - (filterMaps[2] - 1)) / stride[0]))
			filterToeplitz = (filterMaps[0], inputFeatureMaps[0] * filterMaps[2] * filterMaps[3])
			outputToeplitz = (filterMaps[0], inputToeplitz[1])

			computeTime = (filterToeplitz[0] * filterToeplitz[1] * inputToeplitz[1]) / (matrixK * matrixN * matrixM * matrixPrimitivesInParallel * matrixCompletionsPerSec)
			totalComputeTime = computeTime

			outputFeatureMaps = (filterMaps[0], math.ceil((inputFeatureMaps[1] + padding[1] - (filterMaps[2] - 1)) / stride[0]), math.ceil((inputFeatureMaps[2] + padding[0] - (filterMaps[3] - 1))/stride[1]))
			outputMemory = outputFeatureMaps[0] * outputFeatureMaps[1] * outputFeatureMaps[2]
			if outputMemory > freeInternalMemory:
				outputLocation = 0
				outputMovementTime = outputMemory / memoryBandwidth
			else:
				outputLocation = 1
				outputMovementTime = 0
			totalMovementTime = inputMovementTime + outputMovementTime + filterMovementTime

			seriesTotalTime = totalMovementTime + totalComputeTime
			parallelTotalTime = max(totalMovementTime, totalComputeTime)

			output_dims_dict[layer['name']] = OutputDims(outputFeatureMaps[1], outputFeatureMaps[2], outputFeatureMaps[0])

			layer_info_array = [layer['name'], inputLocation, 1, inputFeatureMaps[0], inputFeatureMaps[1], inputFeatureMaps[2], inputMemory, outputLocation, 1, outputFeatureMaps[0], outputFeatureMaps[1], outputFeatureMaps[2], outputMemory, filterLocation, 1, filterGrouping, filterMaps[2], filterMaps[3], filterMemory, inputMovementTime, filterMovementTime, outputMovementTime, totalMovementTime, computeTime, 0, totalComputeTime, seriesTotalTime, parallelTotalTime]
			
			inputLocation = outputLocation


	else:
		# Number of channels
		if layer['config']['batch_input_shape'][3] is not None:
			channels = layer['config']['batch_input_shape'][3]
		else:
			channels = 3

		# Number of rows
		if layer['config']['batch_input_shape'][1] is not None:
			rows = layer['config']['batch_input_shape'][1]
		else:
			rows = 299

		# Number of columns
		if layer['config']['batch_input_shape'][2] is not None:
			columns = layer['config']['batch_input_shape'][2]
		else:
			columns = 299

		inputMemory = channels * rows * columns
		inputLocation = 1
		inputMovementTime = inputMemory / memoryBandwidth
		totalMovementTime = inputMovementTime

		seriesTotalTime = totalMovementTime
		parallelTotalTime = totalMovementTime

		layer_info_array = [layer['name'], 0, 1, channels, rows, columns, inputMemory, 1, 1, channels, rows, columns, inputMemory, 0, 0, 0, 0, 0, 0, inputMovementTime, 0, 0, totalMovementTime, 0, 0, 0, seriesTotalTime, parallelTotalTime]

		output_dims_dict[layer['name']] = OutputDims(rows, columns, channels)

	complete_total_info_array.append(layer_info_array)


data = pd.DataFrame(complete_total_info_array, columns = ["Layer Name",
														  "Input Feature Map Location", 
														  "Input upsampling ratio",
														  "Input channels",
														  "Input rows",
														  "Input columns",
														  "Input bytes",
														  "Output Feature Map Location",
														  "Output down sampling location",
														  "Output channels",
														  "Output rows",
														  "Output columns",
														  "Output bytes",
														  "Filter co-efficient Location",
														  "Filter upsampling ratio",
														  "Filter grouping",
														  "Filter rows",
														  "Filter columns",
														  "Filter bytes",
														  "Input feature map movement time",
														  "Filter co-efficient movement time",
														  "Output feature map data movement time",
														  "Total data movement time",
														  "Matrix compute time",
														  "Vector compute time",
														  "Total compute time",
														  "Serial total time",
														  "Parallel total time"])





data.to_csv(inputJson.split(".")[0] + "_output.csv")

with open(inputJson.split(".")[0] + "_output.csv", "a") as f:
	for i in range(6):
		f.write("\n")

	f.write("Total Input feature map movement time = {0:.2f} ms\n".format(data["Input feature map movement time"].sum() * 1000))
	f.write("Total Filter co-efficient movement time = {0:.2f} ms\n".format(data["Filter co-efficient movement time"].sum() * 1000))
	f.write("Total Output feature map data movement time = {0:.2f} ms\n".format(data["Output feature map data movement time"].sum() * 1000))
	f.write("Total data movement time of network = {0:.2f} ms\n".format(data["Total data movement time"].sum() * 1000))

	f.write("Total Matrix compute time = {0:.2f} ms\n".format(data["Matrix compute time"].sum() * 1000))
	f.write("Total Vector compute time = {0:.2f} ms\n".format(data["Vector compute time"].sum() * 1000))
	f.write("Total compute time of network = {0:.2f} ms\n".format(data["Total compute time"].sum() * 1000))

	f.write("Total Serial total time = {0:.2f} ms\n".format(data["Serial total time"].sum() * 1000))
	f.write("Total Parallel total time = {0:.2f} ms\n".format(data["Parallel total time"].sum() * 1000))