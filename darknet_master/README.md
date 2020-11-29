![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

[anchor_box_generator.py] : this script generates probable anchor boxes based on a few input images present inside backup/train.txt

[plot_logfile_loss.py] : this python script plots the loss during training saved in the log file 

# to save the logs of training
$ ./darknet detector train backup/nfpa.data cfg/yolov3.cfg weights/darknet53.conv.74 >> backup/name.log

# to plot the Loss during training saved inside the log file
$ python3 plot_logfile_loss.py backup/name.log



