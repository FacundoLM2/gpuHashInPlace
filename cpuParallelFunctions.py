from joblib import Parallel, delayed  
import multiprocessing

import numpy as np
import numpy_indexed as npi

def makePropertySlice(sliceImage,compactifyIDs_LUT):
    sliceProperty=(sliceImage).copy().astype(np.uint32)

    output_array = npi.remap(
        sliceProperty.flatten(), 
        list(compactifyIDs_LUT.keys()), 
        list(compactifyIDs_LUT.values())).reshape(sliceImage.shape)

    return output_array

def makePropertyMap(V_fiberMap,marker_to_propertyLUT,parallelHandle=True):

    V_fiberMap_property=np.empty(V_fiberMap.shape,np.float32)
    
    if parallelHandle:
        num_cores=min(int(multiprocessing.cpu_count()-2),48)#will cause memory overload for large sets if too many cores used 
    else:
        num_cores=1

    print("python: launching makePropertyMap on {} CPU threads".format(num_cores))

    results = Parallel(n_jobs=num_cores)\
    (delayed(makePropertySlice)\
        (
            V_fiberMap[iSlice],
            marker_to_propertyLUT
        )for iSlice in range(V_fiberMap.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_fiberMap_property[iSlice]=resTuple

    return V_fiberMap_property