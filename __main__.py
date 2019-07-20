import cv2 as cv
import numpy as np
from pynput.mouse import Button,Controller

from PySide2.QtWidgets import QApplication
from PySide2.QtQuick import QQuickView
from PySide2.QtCore import QUrl


# app = QApplication([])
# view = QQuickView()
# url = QUrl(r"D:\Projects\GUI\view.qml")
# view.setSource(url)
# view.showFullScreen()
# app.exec_()


mouse=Controller()

# Write down conf, nms thresholds,inp width/height
confThreshold = 0.5
nmsThreshold = 0.5
inpWidth = 288
inpHeight = 288

# Load names of classes and turn that into a list
#classesFile = "coco"
classesFile = r"D:\Projects\Licenta\cfg_and_weights\4_classes_416\obj.names"
classes = None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Model configuration
#modelConf = 'yolov3.cfg'
#modelWeights = 'yolov3.weights'

# modelConf = r'D:\Projects\Licenta\Yolo\tiny cfgs\yolov3-tiny.cfg'
# modelWeights = r'D:\Projects\Licenta\Yolo\tiny cfgs\yolov3-tiny.weights'
#modelWeights = 'D:\\Projects\\Licenta\\Yolo\\FirstTry\\yolov3-tiny.weights'


# modelConf = r'D:\Projects\Licenta\cfg_and_weights\2_classes_416\yolov3-tiny.cfg'
# modelWeights = r'D:\Projects\Licenta\cfg_and_weights\2_classes_416\yolov3-tiny_30000.weights'

'''
3 classes 17k images for training
'''

# modelConf = r'D:\Projects\Licenta\cfg_and_weights\3_classes_416\yolov3-tiny.cfg'
# # modelWeights = r'D:\Projects\Licenta\cfg_and_weights\3_classes_416\yolov3-tiny_1000.weights'
# modelWeights = r'D:\Projects\Licenta\cfg_and_weights\3_classes_416\yolov3-tiny_4000.weights'



'''
4 classes 5.1k img correct labels, few errors on anchor calc
5000 is the best working
'''
modelConf = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\yolov3-tiny.cfg'
modelWeights = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\yolov3-tiny_15000.weights'#D:\Projects\Licenta\cfg_and_weights\4_classes_416\VIII9_DATASET_cfg\


'''q
4 classes 1600 img correct labls
VIII9 dataset
7000 looks to work best
'''
modelConf = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\test cfg\yolov3-tiny.cfg'
modelWeights = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\VIII9_DATASET_cfg\yolov3-tiny_11000.weights'#D:\Projects\Licenta\cfg_and_weights\4_classes_416\VIII9_DATASET_cfg\

'''
Siemens Birou DATASET
1600 imgs
'''
modelConf = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\test cfg\yolov3-tiny.cfg'
modelWeights = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\Siemens_Birou_cfg\yolov3-tiny_9000.weights'#D:\Projects\Licenta\cfg_and_weights\4_classes_416\VIII9_DATASET_cfg\

'''
Combined Siemens Birou + VIII9 DATASET
3200 imgs
'''
modelWeights = r'D:\Projects\Licenta\cfg_and_weights\4_classes_416\Siemens_VIII9_cfg\yolov3-tiny_30000.weights'



#C:\Users\Cristi\darknet\data
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width / 2)
                top = int(centerY - height / 2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
        # print("class: {} left: {} top: {}".format(classIDs[i],left,top))

def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        # if  'open' in label:
        #     print("It's working ")
        #     print(left)
    # A fancier display of the label from learnopencv.com
    # Display the label at the top of the bounding box
    # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
    # (255, 255, 255), cv.FILLED)
    # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    mouse.position = (((left+right)), ((top+bottom))-50)
    if 'select' in label:
        mouse.press(Button.left)
    else:
        mouse.release(Button.left)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Process inputs
winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)

# cv.CAP_PROP_FRAME_WIDTH =288
# cv.CAP_PROP_FRAME_HEIGHT=288

cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    # get frame from video
    hasFrame, frame = cap.read()

    # Create a 4D blob from a frame

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Set the input the the net
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    # show the image
    #cv.imshow(winName, frame)

