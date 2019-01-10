#!/bin/bash

DATA_FOLDER="/home/gurbain/docker_sim"

if [ -d "$DATA_FOLDER" ]; then
	docker run \
       --add-host=paard:192.168.0.2 \
       --add-host=geit:192.168.0.5 \
       --add-host=kat:192.168.0.3 \
       --add-host=schaap:192.168.0.1 \
       -e DISPLAY=$DISPLAY \
       -v $DATA_FOLDER:$DATA_FOLDER \
       -it --rm hyq:latest /bin/bash
else
	docker run \
       --add-host=paard:192.168.0.2 \
       --add-host=geit:192.168.0.5 \
       --add-host=kat:192.168.0.3 \
       --add-host=schaap:192.168.0.1 \
       -e DISPLAY=$DISPLAY \
       -it --rm hyq:latest /bin/bash
fi

