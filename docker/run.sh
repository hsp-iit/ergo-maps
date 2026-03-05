#!/bin/bash
NAME=vlm_test
TAG=latest

sudo xhost +
sudo docker run \
     --network=host --privileged \
     -it \
     --rm \
     --gpus all \
     -e DISPLAY=unix${DISPLAY} \
     --device /dev/dri/card0:/dev/dri/card0 \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     ${NAME}:${TAG} bash
