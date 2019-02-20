#!/bin/bash

# NFS server
sudo apt-get install -y nfs-kernel-server
sudo bash -c 'echo "/export          192.168.0.2/24(rw,fsid=0,insecure,no_subtree_check,async)" >> /etc/exports'
sudo bash -c 'echo "/export/gurbain  192.168.0.2/24(rw,nohide,insecure,no_subtree_check,async,all_squash,anonuid=20001,anongid=7149)" >> /etc/exports'
sudo mkdir -p /export/gurbain  && sudo chown -R gurbain:wal-humanbrainpr /export
sudo service nfs-kernel-server restart 