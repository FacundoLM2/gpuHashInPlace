#include <iostream>
#include <fstream>
#include "vector"

#include "gpuInterface.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std; 


//hash table implemented for uint32_t. this offsets the float value so some precision can be 
//kept when converting float to int. (further precision loss is inconsequential)
const float scalarOffset=1E6; 

uint32_t loadMarkersFromDict(const py::dict& inputLUT,std::vector<KeyValue>& insert_kvs){

  uint32_t num_entries=0;

  for (auto it : inputLUT) {
    std::string markerStr=py::cast<std::string>(it.first);
    uint32_t marker = (uint32_t)stoi(markerStr);
    float    value  = py::cast<float>   (it.second); 

    KeyValue tempKeyValue={marker,uint32_t(value*scalarOffset)};
    insert_kvs.push_back({tempKeyValue});

    num_entries++;
  }

  return num_entries;
}



int run(const py::dict& inputLUT,py::array_t<uint32_t> inputArray){
  py::buffer_info buf1 = inputArray.request();

  uint32_t ARRAY_SIZE=buf1.size;

  uint32_t *ptrToNumpyArray = (uint32_t *) buf1.ptr;

  std::vector<KeyValue> insert_kvs,query_kvs;

  uint32_t num_entries=loadMarkersFromDict(inputLUT,insert_kvs);

  cout << "\n\nfirst in insert_kvs: " << insert_kvs.front().key << " , "<< float(insert_kvs.front().value)/scalarOffset << endl;
  cout << "2nd in insert_kvs: " << insert_kvs[2].key    << " , "<< float(insert_kvs[2].value)/scalarOffset <<endl;
  cout << "last in insert_kvs: " << insert_kvs.back().key    << " , "<< float(insert_kvs.back().value)/scalarOffset << "\n\n"<<endl;


  const uint32_t ARRAY_BYTES = ARRAY_SIZE * sizeof(uint32_t);

  uint32_t* h_in=new uint32_t[ARRAY_SIZE];

  for (int i=0;i<ARRAY_SIZE;i++){
    h_in[i]=(uint32_t) ptrToNumpyArray[i];
  }

  cout<<"starting insertion"<<endl;

  KeyValue* pHashTable = create_hashtable(); //TODO add check for hash table size vs marker count

  // Insert items into the hash table
  const uint32_t num_insert_batches = std::min(16,(int)ARRAY_SIZE); // emtpy inserts (for small ARRAY_SIZE) will cause silent crash
  uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    insert_hashtable(
      pHashTable, 
      insert_kvs.data() + i * num_inserts_per_batch, 
      num_inserts_per_batch
    );
  }

  cout << "\n\n lookup directly on array, parallel: \n" << endl;

  uint32_t num_lookup_per_batch = ARRAY_SIZE / num_insert_batches;

  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    lookup_hashtable_on_array(
      pHashTable, 
      &ptrToNumpyArray[i*num_lookup_per_batch], 
      num_lookup_per_batch, 
      num_lookup_per_batch*sizeof(uint32_t)
      );
  }

  destroy_hashtable(pHashTable);

  return scalarOffset;
}

PYBIND11_MODULE(cudaProcesses, m) {
    m.doc() = "cudaProcesses pybind11 example"; // optional module docstring

    m.def("run", &run, "load image, change it and write to disk");
}