#!/bin/bash

CONTAINER_NAME="foundationpose"
docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME /bin/bash