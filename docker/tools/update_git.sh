#!/bin/bash

REMOTE="origin"
BRANCH="master"

DOCKER_IMG="hyq"
DOCKER_TAG="latest"


echo " --- Pull Git Repository Image --- "
docker run -it $DOCKER_IMG:$DOCKER_TAG /bin/sh -c "cd /home/gurbain/hyq_ml && git config --global user.email 'gurbain@ugent.be' \
                                                   &&  git config --global user.name 'Gabriel Urbain' \
                                                   && git stash && git pull $REMOTE $MASTER"

echo " --- Get New Container ID --- "
CONTAINER_ID=$(docker container ls --all --quiet | head -n 1)

echo " --- Commit Docker Image and Kill It ---"
docker commit $CONTAINER_ID $DOCKER_IMG:$DOCKER_TAG
docker container stop $CONTAINER_ID
docker container rm $CONTAINER_ID

