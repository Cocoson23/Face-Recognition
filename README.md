# OpenCV_Face_Detection
## Enviroment  
- Pip  
  - OpenCV
  - Numpy
- Anaconda
- Pycharm

## How to train and run?  
- `mkdir dataset and test_output`  
---project  
&nbsp;&nbsp;|  
&nbsp;&nbsp;---dataset  
&nbsp;&nbsp;|&nbsp;&nbsp;|  
&nbsp;&nbsp;|&nbsp;&nbsp;---face  
&nbsp;&nbsp;|&nbsp;&nbsp;---test  
&nbsp;&nbsp;|&nbsp;&nbsp;|  
&nbsp;&nbsp;---test_output   
- `run image_take.py`  
    Get face data by camera and build dataset.  
---dataset  
&nbsp;&nbsp;|  
&nbsp;&nbsp;---person1  
&nbsp;&nbsp;|  
&nbsp;&nbsp;---person2  
&nbsp;&nbsp;|  
&nbsp;&nbsp;......  
- `run train.py`  
    Edit params and train face detection model.  
- `run detect.py`  
    Use the trained model to detect in real time with the camera.
## How to test?  
- `run test_image_take.py`  
    Get frame by camera and build test dataset.  
- `run test.py`  
    Detect test dataset, and place the output to ./test_output/.  
