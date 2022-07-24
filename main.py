import json
import os
import time
from tifffile import TiffFile
import numpy as np
import matplotlib.pyplot as plt

from cpuParallelFunctions import makePropertyMap
import build.cudaProcesses as cudaProcesses



###############################################

##

print("\npython: loading markersLUT")

with open("marker_to_propertyLUT.json","r") as f:
    inputLUT=json.load(f)

print("python: loading tiff file")

with TiffFile("V_uint16.tiff") as tif:
    V_input=tif.asarray().astype(np.uint32)

sliceForPlot=400

# V_input=V_input[:sliceForPlot+1]

del inputLUT["-1"]

offsetFactor=1e6
inputLUT_python={int(key):int(val*offsetFactor) for key,val in inputLUT.items()}


tic = time.perf_counter()
V_python=makePropertyMap(V_input, inputLUT_python,parallelHandle=True)
V_python=V_python.astype(np.float32)/offsetFactor

toc=time.perf_counter()

time_python=time.strftime("%Mm%Ss", time.gmtime(toc-tic))


inputSlice=V_input[sliceForPlot].copy()

shapeTuple=V_input.shape

V_input=V_input.ravel()


print("\ncalling cudaProcessis.run()")

tic = time.perf_counter()

offsetFactor=cudaProcesses.run(inputLUT,V_input)

toc=time.perf_counter()

time_cuda=time.strftime("%Mm%Ss", time.gmtime(toc-tic))

print("back to python:")

plt.figure(figsize=[8,8],num="inputSlice")
plt.imshow(inputSlice)

V_input=V_input.reshape(shapeTuple)
V_input=V_input/offsetFactor



plt.figure(figsize=[8,8],num="outputSlice_cuda")
plt.imshow(V_input[sliceForPlot])

plt.figure(figsize=[8,8],num="outputSlice_python")
plt.imshow(V_python[sliceForPlot])

plt.figure(figsize=[8,8],num="Error checking: python vs cuda")
plt.imshow(V_python[sliceForPlot]-V_input[sliceForPlot])

print("\n\nExecution time_python : ",time_python)
print(    "Execution time_cuda   : ",time_cuda)

plt.show()