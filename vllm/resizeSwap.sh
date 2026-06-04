#!/bin/bash
#Usage: $1=size in GB

css(){  #change swap size
    sudo swapoff -a
    sudo fallocate -l $1G /swap.img
    sudo mkswap /swap.img
    sudo swapon /swap.img
}

echo $#

if [ $# -eq 1 ]; then
    css $1
else
    echo "usage: bash resizeSwap 64"
    exit;
fi
