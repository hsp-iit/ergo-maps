#!/bin/bash         
cd $PWD
docker build . --build-arg "GIT_USERNAME=$1" --build-arg "GIT_USER_EMAIL=$2" -t vlm_test:latest -f Dockerfile_u24_cu128
