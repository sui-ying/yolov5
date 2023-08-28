#!/bin/bash

nc=$1

wdir=${PWD}

build_path=${wdir}/build/build_${nc}nc
if [ ! -d ${build_path} ];
then
  mkdir ${build_path}
fi

cd ${build_path}
cmake ../..
make -j8
