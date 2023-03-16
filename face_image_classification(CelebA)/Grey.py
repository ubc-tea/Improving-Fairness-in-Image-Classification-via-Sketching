import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--sensitive_type', type=str, default='gender', help='the sensitive attribute, default gender, other type: skin_color, hair_color') 
 
opt = parser.parse_args()


def grey_image(inputpath, outputpath):

    img_list = os.listdir(inputpath)

    for img in img_list:
        I = cv2.imread(inputpath+img)
        IMG_1 = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        IMG_3 = cv2.cvtColor(IMG_1, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(outputpath+img, IMG_3)

inputpath = './dataset/' + opt.sensitive_type + '/origin/train/'
outputpath = './dataset/' + opt.sensitive_type + '/grey/train/'
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
    grey_image(inputpath,outputpath)
    print('finished train grey transfer')
else:
    print('Alredy finished train grey transfer')

inputpath = './dataset/' + opt.sensitive_type + '/origin/valid/'
outputpath = './dataset/' + opt.sensitive_type + '/grey/valid/'
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
    grey_image(inputpath,outputpath)
    print('finished valid grey transfer')

else:
    print('Alredy finished valid grey transfer')


inputpath = './dataset/' + opt.sensitive_type + '/origin/test/'
outputpath = './dataset/' + opt.sensitive_type + '/grey/test/'
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
    grey_image(inputpath,outputpath)
    print('finished test grey transfer')

else:
    print('Alredy finished test grey transfer')



