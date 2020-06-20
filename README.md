# 面向光学成像的水下多目标识别与跟踪系统设计与实现

HEU毕业设计

使用Keras YOLOv3+Kalman Filter算法实现水下多目标识别与跟踪

## 前言
data目录当中，test目录下是测试集，train目录下是训练集。
data目录中还有yolo模型所需的classes.txt、yolo_anchors.txt和train_data.txt。
在pre_train目录当中，有yolo的预训练模型yolov3.weight，以及用convert.py转换的yolo_weights.h5。
在model_data目录中，存储训练结束的模型yolo1.h5、kmeans.py生成的yolo1_anchors.txt、包含holothurian echinus scallop starfish waterweeds5类的voc1_class。(yolo_anchors、voc_class和yolo.h5是第一次训练的含fish/human_face的模型）

## 说明
data、model_data和pre_train内部分文件较大没有上传，即训练集以及预训练模型和训练好的模型。原data目录当中，test目录下是测试集，包括图像增强后的图像，trian目录下是训练集，
也包括图像增强后的图像。data目录中还有yolo模型所需的classes.txt、yolo_anchors.txt和train_data.txt。在pre_train目录当中，有yolo的预训练模型yolov3.weight，
以及用convert.py转换的yolo_weights.h5。在models目录中，存储训练结束的模型和Tensorboard记录。

## 图像增强
首先，对图像采取增强算法增强(image_aug.py)后进行存储，图像增强的算法只是照搬了github上的一个开源。为了满足后续的数据增强需求，**对训练数据进行原图尺寸增强**，但会使得处理速度很慢。为了解决这个问题，尝试采用gpu版的numpy:cupy来调用gpu加速处理，
但增强算法中的有些numpy函数cupy尚未支持，所以只能用numpy，之后采用了10个进程的多进程方式来尽量的加速处理。**对于测试集，直接缩放至目标尺寸**来增强，
这样处理速度就会非常快，避免大尺寸图片耗时的问题，之后再进行输入处理，以满足模型输入需求。具体的处理方式在[数据增强](#数据增强)部分介绍。

## 数据预处理
data_process.py中，对所有图像进行统计，对每一个候选框，进行细致的处理，代码中给出了详细的注释。候选框处理时**删除那些面积小于120的过小候选框**。之后再靠kmeans.py来生成yolo所需的anchors，最后得到data目录中的train_data.txt和yolo_anchors.txt。
另外，赛事的目标种类只有4种，但数据集里有5种，多了一个水草。观察数据集后发现，水草和海胆在某些情况下十分相似，为了使模型增强对水草的区分能力，
**在训练时用5个类别去训练，在预测的时候把水草类丢弃即可**。

## 数据增强
数据增强代码在yolo3目录的utils.py中。

具体步骤：对于训练数据，首先按照7:3的可能性进行**直接缩放或者随机裁剪**。直接缩放过程中删除缩放后面积小于120的过小候选框，
随机裁剪出来的尺寸也与按照原图直接缩放策略缩放后的尺寸相同。之后都是按照1:1的可能性**按顺序进行二次的轻度图像增强，水平翻转和垂直翻转**。对于测试集，直接进行缩放即可。之后按照上述所说的策略填充灰色像素。最后进行模型输入格式的处理，**像素归一化和候选框处理**。

## 模型训练
在train.py中，输入尺寸为416×416，训练集和测试集按照9:1划分。训练过程分为两个阶段，第一阶段冻结预训练所有层，采用RAdam(最小值设为1e-5)， warm_up策略，batch_size设为32，训练100轮，同时做Tensorboard记录。
在第二阶段打开全部网络层来训练，采用RAdam(最小值设为1e-6)，warm_up策略，swa算法，cosine-annealing学习率策略（范围1e-2到1e-6），batch_size设为8（主要原因是显存限制），训练200轮，同时做Tensorboard记录，通过ModelCheckpoint策略对每一轮按照val_loss来决定是否保存模型，最终选用val_loss最小的模型来做预测。将最好的模型更名为yolo1.h5。

## 预测
在yolo.py、yolo_matt.py、yolo_video.py中，代码中的注释很详细，这里不再阐述。
在参数上，**score阈值设为0.001，iou阈值（包括模型阈值和WBF阈值）设为0.25**能得到最高的分数。

yolov3部分的代码是基于[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)进行更改的。

## ps
最终训练出来最好的模型我会放在releases当中。
