#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

#source activate pytorch1.0.0

cd MinDepthFlowProjection
rm -rf build *.egg-info dist 
python setup.py install
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd Interpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..