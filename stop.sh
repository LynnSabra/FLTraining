#!/bin/bash

MODEL=$1

if [ ! -n "$MODEL" ];then
	echo "Please input model"
	exit
fi
ps -ef | grep ${MODEL} | grep -v grep | awk '{print $2}' | xargs kill -9
