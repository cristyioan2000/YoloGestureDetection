# YoloGestureDetection

## This project contains a trained hand model for specific hand gestures and facial recognition.
The fatial recognition is made with the model from opencv dlib.

### This model was trained using darknet https://github.com/pjreddie/darknet


### The training took place on a Google Cloud Ubuntu VM with a Tesla K80 GPU.


## DATASET
The dataset is selfmade following some parameters whom were observed during the creation of previous datasets.
The script used for sampels gathering is here https://github.com/cristyioan2000/yolo_format_gen .
The final model was trained 8000+ hand gesture images. 

## Training
The model was trained in multiple sessiosn  on google colab. Time trained: several days, nearly a weeek. The model is slightly overfitted because the most of the sampels are made using just one person's hand. But it can be retrained or fine-tuned. The model converged around 30.000 iterations.

# Weight link to google drive #
https://drive.google.com/open?id=1t2ROQkOYinSsk9_i5663l2sHCs5_xtQ1


## Results ##
![alt text](https://github.com/cristyioan2000/YoloGestureDetection/blob/master/Res.png)

# How to improve it
There is a better SSD on the market(free), M2DET. You could try that, or use a depth camera in order to gain more accuracy. GPU usage is encouraged, if you have access to a slightly better GPU than your CPU, go for the GPU and use more layers in the model's architecture. 
