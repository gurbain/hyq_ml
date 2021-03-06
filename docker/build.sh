#!/bin/bash
IMAGE_NAME='hyq'
HOSTN="`hostname -f`"


docker stop $IMAGE_NAME
case $HOSTN in
    *"laptop"*)
    echo " === Building on the IDLab LAN === "
    docker build \
        --network="host" \
        --add-host=gu-laptop:127.0.1.1 \
        --add-host=koe:172.18.20.19 \
        --add-host=koe.elis.ugent.be:172.18.20.19 \
        --add-host=paard:172.18.20.20 \
        --add-host=paard.elis.ugent.be:172.18.20.20 \
        --add-host=geit:172.18.20.237 \
        --add-host=geit.elis.ugent.be:172.18.20.237 \
        --add-host=kat:172.18.20.240 \
        --add-host=kat.elis.ugent.be:172.18.20.240 \
        --add-host=hond:172.18.20.241 \
        --add-host=hond.elis.ugent.be:172.18.20.241 \
        --add-host=schaap:172.18.20.236 \
        --add-host=schaap.elis.ugent.be:172.18.20.236 \
        --add-host=nas:172.18.20.252 \
        -t $IMAGE_NAME .
    ;;
    *"iminds.be"*)
    echo " === Building on the IDLab Virtual Wall === "
    docker build \
       --add-host=paard:192.168.0.2 \
       --add-host=geit:192.168.0.5 \
       --add-host=kat:192.168.0.3 \
       --add-host=schaap:192.168.0.1 \
       -t $IMAGE_NAME .
    ;;
    *)
    echo " === Building with no redirection specified === "
    docker build -t $IMAGE_NAME .
    ;;
esac

