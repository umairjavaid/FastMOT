#!/bin/bash
cd /workspace/trtx-fastmot/tensorrtx/yolov5
rm -r build
mkdir build && cd build
cp /workspace/yolov5s.wts .
cmake ..
make
./yolov5 -s yolov5s.wts yolov5s.engine s
#remove already built trt models if any from fastmot repo
cd /workspace/FastMOT/fastmot/models
rm *.trt

cd /workspace/FastMOT
