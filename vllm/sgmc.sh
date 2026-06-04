#!/bin/bash
## suppress gpu memory complains

loop=0
while true;
do
    set +x
    sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches';
    printf "\r\033[K\033[5mSuppressing GPU Memory Complains/${loop}\033[0m"
    loop=$((loop+1))
done
