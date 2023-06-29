**Humman Action Recognition tool**

docker pull ghcr.io/flexigrobots-h2020/har-tool:v0

*video pre-recorded:*

$ docker run -it -e BROKER_PORT=xxxx -e BROKER_PASSWORD="xxx" -e BROKER_USER="xxx" -e BROKER_ADDRESS="xxxx" -v "$(pwd)":/wd/shared --name har ghcr.io/flexigrobots-h2020/har-tool:v0 --video-file shared/input/vendimia_2_recorte.mp4 --buffer-dir ./output/buffer/ --fps-keep 10.0 --video-duration 20 --det-score-thr 0.3 --action-score-thr 0.4 --predict-stepsize 10 --output-stepsize 1 --out-filename shared/output/ --save-vid --save-json --mqtt_output --mqtt_topic common-apps/har-model/output --robot_id tractor_A

*real-time:*

$ sudo chmod a+rw /dev/video0

$ docker run -it -e BROKER_PORT=xxxx -e BROKER_PASSWORD="xxxxx" -e BROKER_USER="xxxxx" -e BROKER_ADDRESS="xxxx" -v "$(pwd)":/wd/shared --device /dev/video0:/dev/video0 --privileged --net=host -v ~/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --name har ghcr.io/flexigrobots-h2020/har-tool:v0 --buffer-dir ./output/buffer/ --fps-keep 10.0 --video-duration 20 --det-score-thr 0.3 --action-score-thr 0.4 --predict-stepsize 10 --output-stepsize 1 --out-filename shared/output/ --save-vid --save-json --mqtt_output --mqtt_topic common-apps/har-model/output --robot_id tractor_A


**Parameters**

- `--buffer-dir`: This is the output directory for raw videos to be processed. It should be a valid path on your system where the program can write data. The default value is the current directory.

- `--fps-keep`: The number of frames per second you want to keep. It should be a floating-point number. The default is 10.0. This means if your video has 30 frames per second, it will only keep every 3rd frame to make it effectively 10 fps.

- `--video-duration`: The duration of the video capture in seconds. It should be an integer, and the default is 30 seconds.

- `--video-file`: An optional pre-recorded video file to be processed instead of real-time camera feed. If specified, the program will analyze this video file instead of capturing video from the camera.

- `--config`: The path to the spatio-temporal detection configuration file. This is a file from the mmaction2 library that contains configurations for detecting actions in the video.

- `--checkpoint`: The path to the checkpoint file for spatio-temporal detection. This is a pre-trained model that the action detection algorithm uses to identify actions.

- `--det-config`: The path to the human detection configuration file from the mmdetection library.

- `--det-checkpoint`: The path to the human detection checkpoint file. This is a pre-trained model that the human detection algorithm uses to identify humans in the video.

- `--det-score-thr`: The threshold of the human detection score. This is a floating-point number, and any detected humans with a score below this threshold will be ignored. The default value is 0.9.

- `--action-score-thr`: The threshold of the human action score. This is a floating-point number, and any detected actions with a score below this threshold will be ignored. The default value is 0.5.

- `--save-vid`: If set, the program will save an output video showing the detected actions.

- `--save-json`: If set, the program will save the detection results in a JSON file.

- `--mqtt_output`: If set, the program will send the output to an MQTT topic.

- `--mqtt_topic`: The name of the MQTT topic to which the output is sent.

- `--robot_id`: The ID of the device. Default value is 'video_A'.

- `--label-map`: The path to the label map file which maps action labels to their IDs.

- `--device`: The computation device to be used. It could be a CPU or CUDA device. The default value is 'cuda:0'.

- `--out-filename`: The name of the output video file. The default value is 'mmaction2/demo/stdet_demo.mp4'.

- `--predict-stepsize`: The step size for giving out a prediction. The program will output a prediction every 'n' frames. The default value is 5.

- `--output-stepsize`: The step size for showing frames in the demo. The program will show one frame every 'n' frames. The default value is 1.

- `--output-fps`: The frames per second of the output video. The default value is 5.

- `--cfg-options`: Overrides some settings in the used configuration file. The key-value pair in the 'xxx=yyy' format will be merged into the configuration file.

Please make sure to replace the paths and filenames with the appropriate ones on your system.

