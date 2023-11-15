# OpenVINO.NET demo for yolov8 detection model

This project is a minimal demo for my [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET) project to infer yolov8 detection model.

## NuGet packages requirements
* Sdcb.OpenVINO
* Sdcb.OpenVINO.runtime.win-x64
* Sdcb.OpenVINO.Extensions.OpenCvSharp4
* OpenCvSharp4
* OpenCvSharp4.runtime.win

## Brief Introduction to Model Inference
The yolov8n model has 80 classifications.

This model has an input size of `1x3x640x640xF32` and an output size of `1x84x8400`.

Here is the output tensor shape explain(from github):

> The first dimension represents the batch size, which is always equal to one.
> The second dimension consists of 84 values, where the first 4 values represent the bounding box coordinates (x, y, width and height) of the detected object, and the rest of the values represent the probabilities of the object belonging to each class.
> Finally, the third dimension represents the maximum number of possible detected objects, which is 8400 in this case.

## Steps to convert from PyTorch yolov8 model into OpenVINO

* Downloaded from [ultralytics official website](https://docs.ultralytics.com/models/yolov8/#supported-modes), specifically, it's `YOLOv8n.pt`(6.23MB).
* Install python, and install `ultralytics`: `pip install ultralytics`
* Convert `YOLOv8n.pt` into OpenVINO xml model via command: `yolo export model=yolov8n.pt format=openvino`
* After convert, you will get `yolov8n.xml`(227KB) and `yolov8n.bin`(12.1MB) in *yolov8n_openvino_model* folder.
