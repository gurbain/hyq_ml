#!/bin/bash

DATA_FOLDER="/home/gurbain/docker_sim"

# docker run \
#        --add-host=koe:172.18.20.19 \
#        --add-host=koe.elis.ugent.be:172.18.20.19 \
#        --add-host=paard:172.18.20.20 \
#        --add-host=paard.elis.ugent.be:172.18.20.20 \
#        --add-host=geit:172.18.20.237 \
#        --add-host=geit.elis.ugent.be:172.18.20.237 \
#        --add-host=kat:172.18.20.240 \
#        --add-host=kat.elis.ugent.be:172.18.20.240 \
#        --add-host=hond:172.18.20.241 \
#        --add-host=hond.elis.ugent.be:172.18.20.241 \
#        --add-host=schaap:172.18.20.236 \
#        --add-host=schaap.elis.ugent.be:172.18.20.236 \
#        --add-host=nas:172.18.20.252 \
#        -e DISPLAY=$DISPLAY \
#        -it hyq:latest /bin/bash

if [ -d "$DATA_FOLDER" ]; then
	docker run \
       --add-host=paard:192.168.0.2 \
       --add-host=geit:192.168.0.5 \
       --add-host=kat:192.168.0.3 \
       --add-host=schaap:192.168.0.1 \
       -e DISPLAY=$DISPLAY \
       -v $DATA_FOLDER:$DATA_FOLDER \
       -it hyq:latest /bin/bash
else
	docker run \
       --add-host=paard:192.168.0.2 \
       --add-host=geit:192.168.0.5 \
       --add-host=kat:192.168.0.3 \
       --add-host=schaap:192.168.0.1 \
       -e DISPLAY=$DISPLAY \
       -it hyq:latest /bin/bash
fi

