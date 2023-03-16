# python libraties
import os, cv2,itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import argparse

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mpl_toolkits.axes_grid1 import make_axes_locatable

# module in need
from module import Discriminator_withoutori, Discriminator_withori, initialize_model
from metrics import SPD, EOD, AOD
import pdb

# Visibility
from tensorboardX import SummaryWriter

matplotlib.use('Agg')

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/', help='the place you store ISIC dataset')
parser.add_argument('--img_type', type=str, default='origin', help='type of image, default origin, other sketch') 
parser.add_argument('--save_dir', type=str, default='./saved_model/', help='the place to store the output models')
parser.add_argument('--gpu_id', type=str, default='0', help='choose the gpu id to use, default = 0')
parser.add_argument('--use_ori', type=bool, default=False, help='choose to use orthogonality structure or not')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate of discriminator, default=1e-4')
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate of generator, default=1e-4')
parser.add_argument('--jcbsize', type=int, default=8, help='size of sub-dimension for computing jacobian matrixs, default=8')
parser.add_argument('--alpha', type=float, default=20, help='weight for orthogonality loss')
parser.add_argument('--delta', type=float, default=0.01, help='step size for computing derivation')
parser.add_argument('--privilege_type', type=str, default='sex', help='choose from sex, age and color')
parser.add_argument('--schema', type=str, default='train', help='turn the model to train mode')
parser.add_argument('--model_path', type=str, default=None, help='load pre-trained model for evaluation, not for train mode')
parser.add_argument('--discriminator_path', type=str, default=None, help='load pre-trained discriminator for evaluation, not for train mode')
parser.add_argument('--seed', type=int, default=1,
                    help='validation fold, choices=[1,2,3,4,5]')
parser.add_argument('--fairloss', type=int, default=1, help='add fairloss to the loss function, default 1')
parser.add_argument('--batch_size', type=int, default=32, help='batch size, default 32')
parser.add_argument('--epochs', type=int, default=20, help='epoch, default 20')
parser.add_argument('--model_name', type=str, default='', help='name of testing model')


opt = parser.parse_args()
print(opt)

EPS = 1e-12
old_loss = 1e12
best_epoch = 0
best_val_acc = 0
index = 0
torch.cuda.empty_cache()
###########################################################################
##########################  Data Preprocessing    #########################
###########################################################################

data_dir = opt.data_dir
if opt.fairloss:
    save_dir = opt.save_dir + opt.privilege_type + '_' + opt.img_type + '_fairloss'
else:
    save_dir = opt.save_dir + opt.privilege_type + '_' + opt.img_type

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))


#df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
datapath = opt.data_dir + opt.img_type + '/'
jpg = '.jpg'
df_original['path'] = datapath + df_original['image_id'] + jpg
#print(df_original['path'].head())
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
#sex bias
sex_type_dict = {
    'male': 0,
    'female': 1
}
#print(df_original['sex'])
df_original['sex_idx'] = pd.Categorical(df_original['sex']).map(sex_type_dict.get)
df_original['sex_idx'] = pd.to_numeric(df_original['sex_idx'], errors='coerce')
df_original = df_original.dropna(subset=['sex_idx'])
df_original['sex_idx'] = df_original['sex_idx'].astype(int)
#print(df_original['sex_idx'].head())

#print(df_original['sex_idx'])

#age bias
#some of the age in the dataset haven't been upload,remove them
#print(df_original[df_original['age'].isnull()])
df_original['age'] = pd.to_numeric(df_original['age'], errors='coerce')
df_original = df_original.dropna(subset=['age'])
df_original['age'] = df_original['age'].astype(int)

def get_age_bias(x):
    if int(x) < 60:
        return 'young'
    if int(x) >= 60:
        return 'old'

df_original['age_idx'] = df_original['age'].apply(get_age_bias)
df_original['age_idx'] = pd.Categorical(df_original['age_idx']).codes   #old:0, young:1
#print(df_original['age_idx'].head())

#color bias
# 1: light skin; 0: dark skin
def get_color_bias(x):
    if int(x) < 32:
        return int(1)
    if int(x) >= 32:
        return int(0)

df_original['color_idx'] = df_original['ita'].apply(get_color_bias)



# this will tell us how many images are associated with each lesion_id
df_undup = df_original.groupby('lesion_id').count()
# now we filter out lesion_id's that have only one image associated with it
df_undup = df_undup[df_undup['image_id'] == 1]
df_undup.reset_index(inplace=True)
#df_undup.head()

# here we identify lesion_id's that have duplicate images and those that have only one image.
def get_duplicates(x):
    unique_list = list(df_undup['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'

# create a new colum that is a copy of the lesion_id column
df_original['duplicates'] = df_original['lesion_id']
# apply the function to this new column
df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
#df_original.head()

df_original['duplicates'].value_counts()

# now we filter out images that don't have duplicates
df_undup = df_original[df_original['duplicates'] == 'unduplicated']
#df_undup.shape

# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
y = df_undup['cell_type_idx']
_, df_val = train_test_split(df_undup, test_size=0.2, random_state=opt.seed, stratify=y)
#df_val.shape

df_val['cell_type_idx'].value_counts()

# This set will be df_original excluding all rows that are in the val set
# This function identifies if an image is part of the train or val set.
def get_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows
# create a new colum that is a copy of the image_id column
df_original['train_or_val'] = df_original['image_id']
# apply the function to this new column
df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
# filter out train rows
df_train = df_original[df_original['train_or_val'] == 'train']


#print('train set length:', len(df_train))
#print('validation set length:', len(df_val))

#print(df_train['cell_type_idx'].value_counts())
#print(df_val['cell_type'].value_counts())

# Copy fewer class to balance the number of 7 classes
data_aug_rate = [15,10,5,50,0,40,5]
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
#print(df_train['cell_type'].value_counts())

# # We can split the test set again in a validation set and a true test set:
# df_val, df_test = train_test_split(df_val, test_size=0.5)
df_train = df_train.reset_index()
df_val = df_val.reset_index()
# df_test = df_test.reset_index()
#print('sex_val:', df_val['sex'].value_counts())
# print('color_val:', df_val['color_idx'].value_counts())

################## MODULE PREPARE #######################
model_name = 'vgg'
num_classes = 7
feature_extract = False # all parameters should be update
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Define the device:
device = torch.device('cuda:'+ opt.gpu_id)
# Put the model on the device:
model = model_ft.to(device)
Tensor = torch.cuda.FloatTensor if device!='cpu' else torch.FloatTensor

norm_mean = (0.49139968, 0.48215827, 0.44653124)
norm_std = (0.24703233, 0.24348505, 0.26158768)
# define the transformation of the train images.
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])

# Define a pytorch dataloader for this dataset, choose the bias from ['age', 'sex', 'color']


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index]) # load image
        y = torch.tensor(int(self.df['cell_type_idx'][index]))  #load label
        
        if opt.privilege_type == 'sex':
            z = torch.tensor(int(self.df['sex_idx'][index]))  #load sex bias
        elif opt.privilege_type == 'age':
            z = torch.tensor(int(self.df['age_idx'][index]))  #load sex bias
        elif opt.privilege_type == 'color':
            z = torch.tensor(int(self.df['color_idx'][index]))  #load color bias

        

        if self.transform:
            X = self.transform(X)

        return X, y, z

# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = HAM10000(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)
# Same for the validation set:
validation_set = HAM10000(df_val, transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=opt.batch_size, shuffle=False, num_workers=8)


# fair loss func
def fairloss(SPD_list, output, bias):
    
    # find prediction
    _, pred = output.max(1)
    #predict = pred.data.cpu().numpy()
    
    # find num of z=0 and z=1
    posz = torch.sum(bias)
    negz = len(bias) - posz

    # find num of y=1z=0 and y=1z=1
    i=0
    y1z0 = 0
    y1z1 = 0
    for i in range (len(bias)):
        if pred[i]==1 and bias[i]==0:
            y1z0+=1
        elif pred[i]==1 and bias[i]==1:
            y1z1+=1

    # calculate SPD
    SPD_score = abs(y1z1/posz - y1z0/negz)
    SPD_list.append(SPD_score)

    # MSE of SPD in each batch
    loss = np.square(np.sum(SPD_list))/len(SPD_list)

    return loss


# Loss function
criterion = nn.CrossEntropyLoss().to(device)
if __name__ == '__main__':
    ####################################################################
    ############################   Model Training     #########################
    ####################################################################
    # For visulization
    writer = SummaryWriter('log')
    if opt.schema == 'train':
        # this function is used during training process, to calculation the loss and accuracy
        class AverageMeter(object):
            def __init__(self):
                self.reset()

            def reset(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0

            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count

        total_loss_train, total_acc_train = [],[]
        def train(train_loader, model, criterion, optimizer, epoch):
            SPD_train_list = []
            model.train()
            train_loss = AverageMeter()
            train_acc = AverageMeter()

            curr_iter = (epoch - 1) * len(train_loader)
            for i, data in enumerate(train_loader):
                images, labels, bias = data
                N = images.size(0)
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)
                #print('label shape', labels.size())
                optimizer.zero_grad()
                outputs = model(images)
                #print('output shape:', outputs.size())
                #print('label shape:', labels.size())
                #print(labels)
                if opt.fairloss:
                    fair_loss = fairloss(SPD_train_list, outputs, bias)
                    loss = criterion(outputs, labels) + fair_loss
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                prediction = outputs.max(1, keepdim=True)[1]
                train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
                train_loss.update(loss.item())

                curr_iter += 1

                
                if (i + 1) % 100 == 0:
                    print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                        epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
                    total_loss_train.append(train_loss.avg)
                    total_acc_train.append(train_acc.avg)

            return train_loss.avg, train_acc.avg

        def validate(val_loader, model, criterion, epoch):
            SPD_val_list = []
            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    images, labels, bias = data
                    N = images.size(0)
                    images = Variable(images).to(device)
                    labels = Variable(labels).to(device)
                    outputs = model(images)
                    fair_loss = fairloss(SPD_val_list, outputs, bias)
                    loss = criterion(outputs, labels) + fair_loss
                    prediction = outputs.max(1, keepdim=True)[1]

                    val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
                    val_loss.update(loss.item())
                    #val_loss.update(criterion(outputs, labels).item())

            print('------------------------------------------------------------')
            print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
            print('------------------------------------------------------------')
            return val_loss.avg, val_acc.avg

        print('Start Training')
        epoch_num = opt.epochs
        best_val_acc = 0
        total_loss_val, total_acc_val = [],[]
        for epoch in range(1, epoch_num+1):
            
            

            if epoch <= 10:
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
            else:
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

            loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
            loss_val, acc_val = validate(val_loader, model, criterion, epoch)
            total_loss_val.append(loss_val)
            total_acc_val.append(acc_val)
            
            if loss_val < old_loss:
                #best_val_acc = acc_val
                print('*****************************************************')
                print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
                print('*****************************************************')
                torch.save(model.state_dict(), os.path.join(save_dir, 'isic_{}_iter{}_accuaracy{}.pt'.format(model_name,str(epoch),str(round(acc_val, 2)))))
                print('Saved a better model')
                print('*****************************************************')
                old_loss = loss_val
                best_epoch = epoch
                best_val_acc = acc_val
                index = 0
            else:
                index+=1

            if index == 5:
                print('Early Stop!')
                print('total epoch trained: %d, best epoch: %d' % (epoch, best_epoch))
                break

        print('Finished Training')

        print('*****************************************************')
        print('The Best Record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (best_epoch, old_loss, best_val_acc))
        print('*****************************************************')

    ###########################################################################
    ############################   Model Evaluation     #######################
    ###########################################################################
    else:
        print("Using existing trained model")
        model.load_state_dict(torch.load(save_dir + '/' + opt.model_name))
        model.eval()
        #if opt.discriminator_path is not None:
        #    discriminators.load_state_dict(torch.load(opt.discriminator_path))
        #    discriminators.eval()

        y_label = []
        y_predict = []
        biases = []
        P_list = []
        SPD_list = []
        EOD_list = []
        AOD_list = []

        with torch.no_grad():

            for i, data in enumerate(val_loader):
                images, labels, bias = data
                N = images.size(0)
                images = Variable(images).to(device)
                outputs = model(images)           
                    

                prediction = outputs.max(1, keepdim=True)[1]
                SPD_score = SPD(prediction.cpu(), bias.cpu(), labels.cpu())
                print('the SPD score is:', SPD_score)
                EOD_score = EOD(prediction.cpu(), bias.cpu(), labels.cpu())
                AOD_score = AOD(prediction.cpu(), bias.cpu(), labels.cpu())
                #print(SPD_score, EOD_score, AOD_score)
                #SPD_score = abs(SPD_score)
                SPD_list.append(SPD_score)
                EOD_list.append(EOD_score)
                AOD_list.append(AOD_score)
                #print(SPD_list)
                
                y_label.extend(labels.cpu().numpy())
                biases.extend(bias.cpu().numpy())
                y_predict.extend(np.squeeze(prediction.cpu().numpy().T))
        
        if opt.discriminator_path is not None:
            print(P_list)
            print(SPD_list)
            #plt.axis([0, 3, 0, 3])
            x = np.arange(3)
            #plt.plot(x, x)
            R2 = r2_score(P_list, SPD_list)
            print('the R2 value is:', R2)
            plt.scatter(SPD_list, P_list)
            plt.xlabel('the ground truth SPD value')
            plt.ylabel('the predict P value')
            plt.savefig('result.jpg')

        #print(SPD_list)
        #P_value = mean(P_list)
        SPD_value = mean(SPD_list)
        EOD_value = mean(EOD_list)
        AOD_value = mean(AOD_list)
        #print('the P value is:', P_value)
        print('the SPD value is:', SPD_value, 
            'the EOD value is:', EOD_value,
            'the AOD value is:', AOD_value)


        #################################################
        ############## female:0 male:1, #################
        #################################################

        def plot_confusion_matrix(cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            plt.figure(figsize=(5,4.3))
            ax = plt.gca()
            #im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            # plt.title(title)
            #plt.colorbar(im, fraction = 0.046, pad = 0.04)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            if normalize:
                cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
        
        # Generate a classification report

        #print(len(y_predict), y_predict)
        #print(len(biases), biases)
        #depart the result to bias and unbias part
        #y_label, biases, y_predict = np.array(y_label), np.array
        y_label_0 = []  
        y_predict_0 = []
        y_label_1 = []  
        y_predict_1 = []
        for i in range(len(y_label)):
            if biases[i] == 0:
                y_label_0.append(y_label[i]) 
                y_predict_0.append(y_predict[i])
            if biases[i] == 1:
                y_label_1.append(y_label[i]) 
                y_predict_1.append(y_predict[i])
        
        print(len(y_label_0))
        print(len(y_predict_0))
        print(len(y_label_1))
        print(len(y_predict_1))

        if opt.privilege_type == 'sex':
            report0 = classification_report(y_label_0, y_predict_0, target_names=plot_labels, digits=3)
            print('the report of male:')
            print(report0)
            report1 = classification_report(y_label_1, y_predict_1, target_names=plot_labels, digits=3)
            print('the report of female:')
            print(report1)
        
        if opt.privilege_type == 'age':
            report0 = classification_report(y_label_0, y_predict_0, target_names=plot_labels, digits=3)
            print('the report of old(more than 60):')
            print(report0)
            report1 = classification_report(y_label_1, y_predict_1, target_names=plot_labels, digits=3)
            print('the report of young(less than 60):')
            print(report1)
        
        if opt.privilege_type == 'color':
            report0 = classification_report(y_label_0, y_predict_0, target_names=plot_labels, digits=3)
            print('the report of low ita value:')
            print(report0)
            report1 = classification_report(y_label_1, y_predict_1, target_names=plot_labels, digits=3)
            print('the report of high ita value:')
            print(report1)