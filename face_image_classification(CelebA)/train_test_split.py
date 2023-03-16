import csv
import argparse
import os
import pandas as pd
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--csv_dir', type=str, default='./dataset/list_attr_celeba.csv', help='the place you store csv') 
parser.add_argument('--data_dir', type=str, default='./dataset/', help='the place you store dataset') 
parser.add_argument('--img_dir', type=str, default='./dataset/img_align_celeba/', help='the place you store images, other root: ./dataset/sketch_celeba/, ./dataset/grey_celeba/') 
parser.add_argument('--img_type', type=str, default='origin', help='type of stored images, default origin, other type: grey, sketch') 
parser.add_argument('--sensitive_type', type=str, default='gender', help='the sensitive attribute, default gender, other type: skin_color, hair_color') 
parser.add_argument('--num', type=int, default=10000, help='the total number of images in the dataset, default 10000') 

opt = parser.parse_args()

Read = open(opt.csv_dir, "r")
reader = csv.reader(Read)
line = Read.readline()

if not os.path.exists(opt.data_dir+opt.sensitive_type):
    os.makedirs(opt.data_dir+opt.sensitive_type)

filename = opt.data_dir+opt.sensitive_type+'/'+opt.sensitive_type

if not os.path.isfile(filename+'_train.csv'):

    print('start csv train test split')

    Wtrain = open(filename+'_train.csv', 'w', newline='')
    train = csv.writer(Wtrain)

    Wvalid = open(filename+'_valid.csv', 'w', newline='')
    valid = csv.writer(Wvalid)

    Wtest = open(filename+'_test.csv', 'w', newline='')
    test = csv.writer(Wtest)

    #fileHeader = ["Image_id", "smile", "male"]
    fileHeader = ["Image_id", "smile", "attractive", opt.sensitive_type]

    train.writerow(fileHeader)
    valid.writerow(fileHeader)
    test.writerow(fileHeader)

    # black hair 9
    # male 21
    # pale skin 27
    # smile 32
    # Attractive 3

    pos=0
    neg=0
    half_num = opt.num * 0.5

    for line in reader:
        id = line[0]
        if opt.sensitive_type == 'gender':
            Sensitive_property = line[21]
        elif opt.sensitive_type == 'skin_color':
            Sensitive_property = line[27]
        elif opt.sensitive_type == 'hair_color':
            Sensitive_property = line[9]

        smile = line[32]
        attractive = line[3]
        #print(male)

        # train
        if (pos<half_num*0.7):
            if (Sensitive_property == '1'):
                train.writerow([id, smile, attractive, Sensitive_property])
                pos+=1

        if (neg<half_num*0.7):
            if (Sensitive_property == '-1'):
                train.writerow([id, smile, attractive, Sensitive_property])
                neg+=1

        # valid
        if (half_num*0.7<=pos<half_num*0.85):
            if (Sensitive_property == '1'):
                valid.writerow([id, smile, attractive, Sensitive_property])
                pos+=1

        if (half_num*0.7<=neg<half_num*0.85):
            if (Sensitive_property == '-1'):
                valid.writerow([id, smile, attractive, Sensitive_property])
                neg+=1

        # test
        if (half_num*0.85<=pos<half_num):
            if (Sensitive_property == '1'):
                test.writerow([id, smile, attractive, Sensitive_property])
                pos+=1

        if (half_num*0.85<=neg<half_num):
            if (Sensitive_property == '-1'):
                test.writerow([id, smile, attractive, Sensitive_property])
                neg+=1

        if (pos==half_num and neg==half_num):
            break

    Wtrain.close()
    Wvalid.close()
    Wtest.close()

    df = pd.read_csv(filename+'_train.csv')
    df = df.replace(-1, 0)
    df.to_csv(filename+'_train.csv', index=False)

    df = pd.read_csv(filename+'_valid.csv')
    df = df.replace(-1, 0)
    df.to_csv(filename+'_valid.csv', index=False)

    df = pd.read_csv(filename+'_test.csv')
    df = df.replace(-1, 0)
    df.to_csv(filename+'_test.csv', index=False)



    print('finished csv train test split')
    # print(pos)
    # print(neg)

else:
    print('Already finished csv train test split')



Rtrain = open(filename+'_train.csv', 'r')
train = csv.reader(Rtrain)
line = Rtrain.readline()

Rvalid = open(filename+'_valid.csv', 'r')
valid = csv.reader(Rvalid)
line = Rvalid.readline()

Rtest = open(filename+'_test.csv', 'r')
test = csv.reader(Rtest)
line = Rtest.readline()

if os.path.exists(opt.data_dir + opt.sensitive_type + '/' + opt.img_type):
    print('Already finished images train test split')

else:
    for line in train:
        id = line[0]
    
        # Source path 
        source = opt.img_dir
        oldname= source+id

        # Destination path
        
        destination = opt.data_dir + opt.sensitive_type + '/' + opt.img_type +'/train/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        newname = destination+id

        copyfile(oldname, newname)

    for line in valid:
        id = line[0]
    
        # Source path 
        source = opt.img_dir
        oldname= source+id

        # Destination path 
        destination = opt.data_dir + opt.sensitive_type + '/' + opt.img_type +'/valid/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        newname = destination+id

        copyfile(oldname, newname)

    for line in test:
        id = line[0]
    
        # Source path 
        source = opt.img_dir
        oldname= source+id

        # Destination path 
        destination = opt.data_dir + opt.sensitive_type + '/' + opt.img_type +'/test/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        newname = destination+id

        copyfile(oldname, newname)

print('finished images train test split')
print('done')