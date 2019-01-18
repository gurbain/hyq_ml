#!/bin/bash
IMAGE_NAME='hyq'
DATA_FOLDER="/home/gurbain/docker_sim"
HOSTN="`hostname -f`"


docker stop $IMAGE_NAME
case $HOSTN in
    *"laptop"*)
        if [ -d "$DATA_FOLDER" ]; then
            echo " === Running on the IDLab LAN with $DATA_FOLDER mounted === "
            docker run \
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
               -e DISPLAY=$DISPLAY \
               -v $DATA_FOLDER:$DATA_FOLDER \
               -it --rm hyq:latest /bin/bash
        else
            echo " === Running on the IDLab LAN with no mount === "
            docker run \
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
               -e DISPLAY=$DISPLAY \
               -it --rm hyq:latest /bin/bash
        fi
    ;;
    *"iminds.be"*)
        if [ -d "$DATA_FOLDER" ]; then
            echo " === Running on the Virtual Wall with $DATA_FOLDER mounted === "
            docker run \
               --add-host=paard:192.168.0.2 \
               --add-host=geit:192.168.0.5 \
               --add-host=kat:192.168.0.3 \
               --add-host=schaap:192.168.0.1 \
               -e DISPLAY=$DISPLAY \
               -v $DATA_FOLDER:$DATA_FOLDER \
               -it --rm hyq:latest /bin/bash
        else
            echo " === Running on the Virtual Wall with no mount === "
            docker run \
               --add-host=paard:192.168.0.2 \
               --add-host=geit:192.168.0.5 \
               --add-host=kat:192.168.0.3 \
               --add-host=schaap:192.168.0.1 \
               -e DISPLAY=$DISPLAY \
               -it --rm hyq:latest /bin/bash
        fi
    ;;
    *)
        if [ -d "$DATA_FOLDER" ]; then
            echo " === Running with no redirection but $DATA_FOLDER mounted === "
            docker run \
               --add-host=paard:192.168.0.2 \
               --add-host=geit:192.168.0.5 \
               --add-host=kat:192.168.0.3 \
               --add-host=schaap:192.168.0.1 \
               -e DISPLAY=$DISPLAY \
               -v $DATA_FOLDER:$DATA_FOLDER \
               -it --rm hyq:latest /bin/bash
        else
            echo " === Running with no redirection and no mount === "
            docker run \
               --add-host=paard:192.168.0.2 \
               --add-host=geit:192.168.0.5 \
               --add-host=kat:192.168.0.3 \
               --add-host=schaap:192.168.0.1 \
               -e DISPLAY=$DISPLAY \
               -it --rm hyq:latest /bin/bash
        fi
    ;;
esac

