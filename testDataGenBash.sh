#!/bin/bash                                                                     
# generate data                                    
for i in {0..7}; do 
  cp data/testData/0 data/8TestData/"$i"
done


# generate MBR files
for i in {0..7}; do 
  cp data/testDataMBRs/0 data/8TestDataMBRs/"$i"
done