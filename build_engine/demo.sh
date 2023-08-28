#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Input parm example as follow: "
    echo "bash demo.sh item "
    exit 1
fi

# $0, 1st Parma is “demo.sh”
flag1=$1
echo "${flag1}"
build_path=build/build_$flag1

#python gen_wts.py --w ./weights/yolov5s.pt  --o ./wts/

#NUM_CLASS=$2
#new_str="CLASS_NUM = $NUM_CLASS; //"
#echo $new_str
#sed "s! CLASS_NUM = ! $new_str ! g" yololayer.h


if [ ! -d "${build_path}" ]; then
  mkdir -p "${build_path}"
else
  cd ${build_path}
  rm -rf *
fi

cd ${build_path}

cmake ../../
make

#./yolov5 -s ../../yolov5-5.0/plate_recog.wts recog_test.engine s


