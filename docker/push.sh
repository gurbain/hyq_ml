#!/bin/bash

DOCKER_IMG="gym_hyq"
DOCKER_TAG="latest"

HUB_IMG="gurbain/gym_hyq"
HUB_TAG="latest"

docker login
docker tag $DOCKER_IMG $HUB_IMG:$HUB_TAG
docker push $HUB_IMG:$HUB_TAG