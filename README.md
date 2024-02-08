# Chess Detection (and tracking)

The goal is, given a top-down video of a chess game, detect and track the board itself as well as the pieces (without Deep-Learning/Neural-Netowork methods, only classical Computer Vision techniques)

## Methodology 
After some experimentation, the method we settled on was:
- for chessboard detection: <i>Harris Corner</i> detection + <i>Hough Lines</i> detection with some fancy postprocessing
- for chess pieces detection: <i>Hough Circles</i> with rudimentary hyperparameter autodetermination
- for chess piece classification attempt: <i>template matching</i> at detected centers to a database of images (to be extended for better results)

You can observe the process in more details with intermediate results in the `tracking_experiments.ipynb` notebook

## Result Example

The detected elements are the chessboard itself, as well as the piece centers with attempted classification by piece color/type. Also, there is preliminary event detection with the detected events written in the top-left (but its still in beta, as you can see)

![result_segment](./readme_resources/ChangingLights3_result_segment.gif)


## Usage
```
python -m chess_detection.video_processing 
    ./my_chess_video.mp4 
    ./my_chess_video_result.mp4 
    --start-time 10 
    --end-time 50 
    --redetect_seconds 0.5
```