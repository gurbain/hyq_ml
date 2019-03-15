#!/bin/bash

DOCKER_IMG="gym_hyq"
DOCKER_TAG="latest"

HUB_IMG="gurbain/gym_hyq"
HUB_TAG="latest"


docker login
docker pull $HUB_IMG:$HUB_TAG
docker tag $HUB_IMG:$HUB_TAG $DOCKER_IMG