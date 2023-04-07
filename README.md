# DDS-V2

This is the implementation for the paper [Server-Driven Video Streaming for Deep Learning Inference](https://kuntaidu.github.io/assets/doc/DDS.pdf).

**Project has been updated to support pytorch based YOLO models to detect objects.
You can find CeyMo pre-trained yolo model file with this project. We used latest YOLO v8 for generate the model.**

Find more details about YOLOv8 (https://github.com/ultralytics/ultralytics)

## 1. Related resources

Please check [Kuntai Du's home page](https://kuntaidu.github.io/aboutme.html) for more DDS-related resources.


## 2. Install Instructions

To run our code, please make sure that conda is installed. Then, under dds repo, run

```conda env create -f conda_environment_configuration.yml```

to install dds environment. Note that this installation assumes that you have GPU resources on your machine. If not, please edit ```tensorflow-gpu=1.14``` to ```tensorflow=1.14``` in ```conda_environment_configuration.yml```.

Now run

```conda activate dds-v2```

to activate dds environment, and 

```cd workspace```

and run 

```wget people.cs.uchicago.edu/~kuntai/frozen_inference_graph.pb```

to download the object detection model (FasterRCNN-ResNet101).

## 3. Run our code

Under ```DDSrepo/workspace```, run

```python entrance.py```

to run DDS!

## 4. Get performance numbers

Under ```DDSrepo/workspace```, run

```python examine.py trafficcam_1 results stats.csv```

you should see something like

```
trafficcam_1_dds_0.8_0.8_36_26_0.0_twosides_batch_15_0.5_0.3_0.01 1055KB 0.857
trafficcam_1_mpeg_0.8_26 3314KB 0.889
trafficcam_1_mpeg_0.8_36 1049KB 0.845
```

The number might vary by platform.

## 5. Some details

If you are considering building your projects based on our codebase, here are some details.

### 5.1 Run in implementation mode

Implementation means we run DDS under real network environment through HTTP post. To do so, in ```DDSrepo```, run

```FLASK_APP=backend/backend.py flask run --port=5001```

and copy the ```frozen_inference_graph.pb``` to ```DDSrepo``` to help the server find the model.

Then use another terminal, cd to ```DDSrepo/workspace```, and edit the mode to ```implementation``` and edit the hname to ```ip:5001``` (ip should be 127.0.0.1 if you run the server locally) to run DDS on implementation mode. You can also run other methods in implementation mode by changing the default value of mode to ```implementation```. 


### 5.2 Inside workspace folder

Inside workspace folder, we use a configuration file ```configuration.yml``` to control the configuration for both the client and the server. This file will be only loaded **once** inside the whole ```python entrance.py``` process. You can add new keys and values in this file. We even support caching, parameter sweeping, and some fancy functionalities in this file. Please read the comments inside this file to utilize it.


## 6. Dataset

### 6.1 Detection dataset

#### 6.1.1 Traffic camera videos and dash camera videos.

We search some keywords through youtube in the anonymous mode of Chrome. The top-ranked search results, corresponding URLS are listed below. We filter out some of these videos.

| Keyword | Source       | Type       | URL                                           | Why we filter it out |
|--------|--------------| ---------- | --------------------------------------------- | -------------------- |
|        |              |            |                                               |                      |
| CeyMo  | Google Drive | dashcam    | <https://drive.google.com/drive/folders/1cjlMDGeM4twNo33959_urmiL3gKx36jC> |                      |

## 7. Additional Notes

### 7.1 Image Preparation

1. To generate images from a video we use ffmpeg multimedia framework. (For More Details about https://ffmpeg.org/about.html)
2. To generate images use following command. (**$ ffmpeg -i raw_footage_1_colombo.mp4 -r 1 -f images %10d.jpg**)
3. To reduce the size larger images, we use imagemagick software. (For More Details about https://imagemagick.org/index.php)
4. To resize the images in a folder use following command. (**$ for image in *.jpg ;  do convert -resize 640x360 "$image" "resized/${image%.*}" ; done**)
5. To convert jpg file format to png we again use imagemagick software. Use following command to convert jpg images to png. (**$ for image in *.jpg ;  do convert "$image" "${image%.*}.png" ; done**) 