#!/bin/bash

REMOTE="origin"
BRANCH="gym-rl"

DOCKER_IMG="gym_hyq"
DOCKER_TAG="latest"

HUB_IMG="gurbain/gym_hyq"
HUB_TAG="latest"

echo " --- Pull Git Repository Image --- "
docker run  $DOCKER_IMG:$DOCKER_TAG /bin/sh -c "cd /home/gurbain/hyq_ml && git config --global user.email 'gurbain@ugent.be' \
                                                   &&  git config --global user.name 'Gabriel Urbain' \
                                                   && git stash && git pull $REMOTE $BRANCH"

echo " --- Get New Container ID --- "
CONTAINER_ID=$(docker container ls --all --quiet | head -n 1)

echo " --- Commit Docker Image and Kill It ---"
docker commit $CONTAINER_ID $DOCKER_IMG:$DOCKER_TAG
docker container stop $CONTAINER_ID
docker container rm $CONTAINER_ID

echo " --- Push New Docker Image on Docker Hub ---"
docker login
docker tag $DOCKER_IMG:$DOCKER_TAG $HUB_IMG:$HUB_TAG
docker push $HUB_IMG:$HUB_TAG