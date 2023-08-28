##### 运行

mycoco下，准备好all_images和all_xml

```
python make_txt.py
```

train_val.py修改class[ ]

```
python train_val.py
```

yolov5-5.0下，修改分类数量和类的名称。打开/data/coco128.yaml

```
# Classes
nc: 7  # number of classes
names: ["wordh","numhl","numhs","wordv","numvl","numvs","numhsq"]
```

打开models/yolo5s.yaml

```
nc: 7  # number of classes
```


# test.py 

训练完成后，复制yolov5/runs/exp下best.pth 到 yolov5-5.0/weights/

'python detect.py'



##### Tensorrt加速yolov5

1. yolov5-5.0文件夹下,pt模型转wts文件

   ```
   python gen_wts.py --w ./weights/yolov5s.pt  --o .
   ```

2. build_engine文件夹下，修改CMakelist.txt

   ```
   # tensorrt
   include_directories(/home/lyz/TensorRT-8.0.0.3/include/)
   link_directories(/home/lyz/TensorRT-8.0.0.3/targets/x86_64-linux-gnu/lib/)
   ```

   编译

   ```
   cmake ..
   make
   ```

3. (build文件夹下）wts转engine 其中： 

   **./yolov5-s** 	运行yolov5.cpp文件 前半部分

   **../../yolov5-5.0/plate_recog.wts** 	wts文件路径

   **recog_test.engine** 	定义生成的engine文件名

   输出到build_engine/build/目录下, build_engine/yololayer.h下 class类要改为训练集一致。
   
   **s** 	yolov5的模式（s/m/l/x）

```
./yolov5 -s ../../yolov5-5.0/plate_recog.wts recog_test.engine s
```

4. 运行engine

   **./yolov5-d**	运行yolov5.cpp文件 后半部分

   **recog_test.engine**  	engine文件名

   **../samples** 	数据集文件名

```
./yolov5 -d recog_test.engine ../samples
```

