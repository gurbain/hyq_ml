#!/bin/bash
set -e

source /opt/ros/kinetic/setup.bash
source /opt/ros/dls-distro/setup.bash
export PATH="/home/gurbain/hyq_ml/docker/bin:$PATH"

exec "$@"