#!/bin/bash

echo "Checking if image is already installed..."
if [[ "$(docker images -q gp-image:latest 2> /dev/null)" == "" ]]; then
  docker build -t gp-image:latest .
fi


if [ ! "$(docker ps -q -f name=gp)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=gp)" ]; then
        # cleanup
        docker rm gp
    fi
    docker run -it -p 4200:4200 --name gp gp-image:latest
fi