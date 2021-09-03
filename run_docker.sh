#!/bin/bash
xhost +

docker run --gpus all -it --rm -e DISPLAY=$DISPLAY --net=host -v $(pwd):/workspace -w /workspace umair:latest 