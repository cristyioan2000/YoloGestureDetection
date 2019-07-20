import os



path = 'build/darknet/x64/data/obj/'
windows_path = 'data\\obj\\'

# Windows
# imgList = os.listdir(r'C:\Users\Cristi\Desktop\darknet-master\darknet-master\build\darknet\x64\data\obj')
# imgList = os.listdir(r'C:\Users\Cristi\Desktop\darknet-master\darknet-master\build\darknet\x64\data\obj')

#Linux
imgList = os.listdir(r'D:\Projects\Licenta\Siemens_birou\obj')
textFile = open('train.txt','w')
for img in imgList:
    if 'jpg' in img.split('.')[1]:
        #Linux
        imgPath = path+img+'\n'

        #Windows
        # imgPath = windows_path+img+'\n'
        imgPath = imgPath.rstrip('\r')
        textFile.write(imgPath)
