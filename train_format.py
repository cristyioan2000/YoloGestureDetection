import os


# Path to Linux
path = 'build/darknet/x64/data/obj/'
# Path to Windows // JUST FOR WINDOWS TRAINING
windows_path = 'data\\obj\\'

# Windows
# imgList = os.listdir(r'C:\Users\Cristi\Desktop\darknet-master\darknet-master\build\darknet\x64\data\obj')
# imgList = os.listdir(r'C:\Users\Cristi\Desktop\darknet-master\darknet-master\build\darknet\x64\data\obj')

# Path to obj (train img paths and labels file )
file_list_path = r''
#Linux
imgList = os.listdir(file_list_path)
textFile = open('train.txt','w')
for img in imgList:
    if 'jpg' in img.split('.')[1]:
        #Linux
        imgPath = path+img+'\n'

        #Windows
        # imgPath = windows_path+img+'\n'
        imgPath = imgPath.rstrip('\r')
        textFile.write(imgPath)
