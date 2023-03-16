# python libraties
import numpy as np
import os
import pandas as pd
from skimage import io
import argparse
from timeit import default_timer as timer

# pytorch libraries
import torch
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision.models as models
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./dataset/', help='the place you store dataset') 
parser.add_argument('--target', type=str, default='smile', help='target label, defalut smile, other target: attractive') 
parser.add_argument('--sensitive_type', type=str, default='gender', help='the sensitive attribute, default gender, other type: skin_color, hair_color') 
parser.add_argument('--img_type', type=str, default='origin', help='type of stored images, default origin, other type: grey, sketch') 
parser.add_argument('--batch_size', type=int, default=64, help='batch size, defalut 64') 
parser.add_argument('--max_epochs_stop', type=int, default=5, help='max_epochs_stop, defalut 3') 
parser.add_argument('--num_epochs', type=int, default=20, help='num_epochs, defalut 10') 
parser.add_argument('--fairloss', type=int, default=1, help='add fairloss to the loss function, default 1')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate, default 1e-3')
parser.add_argument('--isTrain', type=int, default=1, help='1 for train, 0 for test')

opt = parser.parse_args()

torch.cuda.empty_cache()

###########################################################################
##########################  Data Preprocessing    #########################
###########################################################################


class CelebADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) #10000 in this case

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        # 1=Smiling; 2=Attractive
        if opt.target == 'smile':
            y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        elif opt.target == 'attractive':
            y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        bias = torch.tensor(int(self.annotations.iloc[index, 3]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label, bias)
    
train_root_dir = opt.data_dir + opt.sensitive_type + '/' + opt.img_type + '/train'
valid_root_dir = opt.data_dir + opt.sensitive_type + '/' + opt.img_type + '/valid'
test_root_dir = opt.data_dir + opt.sensitive_type + '/' + opt.img_type + '/test'

train_csv = opt.data_dir + opt.sensitive_type + '/'+ opt.sensitive_type +'_train.csv'
valid_csv = opt.data_dir + opt.sensitive_type + '/'+ opt.sensitive_type +'_valid.csv'
test_csv = opt.data_dir + opt.sensitive_type + '/' + opt.sensitive_type +'_test.csv'

train_set = CelebADataset(
    csv_file=train_csv,
    root_dir=train_root_dir,
    transform=transforms.ToTensor(),
)

valid_set = CelebADataset(
    csv_file=valid_csv,
    root_dir=valid_root_dir,
    transform=transforms.ToTensor(),
)

test_set = CelebADataset(
    csv_file=test_csv,
    root_dir=test_root_dir,
    transform=transforms.ToTensor(),
)

test_num = len(test_set)

batch_size = opt.batch_size

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)


###########################################################################
##########################  Model Training        #########################
###########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=opt.max_epochs_stop,
          n_epochs=opt.num_epochs,
          print_every=1
          ):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_best_acc = 0
    
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()

        start = timer()
        
        SPD_train_list = []

        # Training loop
        for ii, (data, target, bias) in enumerate(train_loader):
            
            # Tensors to gpu, both model parameters, data, and target need to be tensors.

            model = model.to(device)
            data = data.to(device)
            target = target.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward path
            output = model(data)
            #print('output:',nn.functional.softmax(output, dim=1))

            # Loss function 
            if opt.fairloss:
                fair_loss = fairloss(SPD_train_list, output, bias)
                loss = criterion(output,target) + fair_loss
            else:
                loss = criterion(output,target)
            #print('fair loss:',fair_loss)


            # Backward path (backpropagation)

            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            #print('train loss:', train_loss)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            #print('pred',pred.data.cpu().numpy())
            correct_tensor = pred.eq(target.data.view_as(pred))

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            SPD_val_list = []

            # Validation loop
            for data, target, bias in valid_loader:
                # Tensors to gpu

                model = model.to(device)
                data = data.to(device)
                target = target.to(device)

                # Forward path

                output = model(data)

                # Validation loss computation
                if opt.fairloss:
                    fair_loss = fairloss(SPD_val_list, output, bias)
                    loss = criterion(output,target) + fair_loss
                else:
                    loss = criterion(output,target)

                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))

                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)


        # Calculate average losses and Calculate average accuracy
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        train_acc = train_acc / len(train_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset)

        history.append([train_loss, valid_loss, train_acc, valid_acc])

        # Print training and validation results
        if (epoch + 1) % print_every == 0:
            print(
                f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
            )
            print(
                f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
            )

        #print((valid_best_acc-valid_acc),(valid_loss-valid_loss_min))
        #print('LOSS DIFFERENCE:',(valid_loss-valid_loss_min))

        # Save the model if validation loss decreases
        # (valid_best_acc-valid_acc) <= 0.03
        if valid_loss < valid_loss_min:
            # Save model 
            # You can use torch.save()


            if ((valid_best_acc-valid_acc) <= 0.03) or (valid_loss <= valid_loss_min):
                torch.save(model.state_dict(), save_file_name)
                print('NEW MODEL SAVED!!!')

                # Track improvement
                epochs_no_improve = 0
                
                valid_loss_save = valid_loss
                valid_acc_save = valid_acc
                #valid_max_acc = valid_acc
                best_epoch = epoch
                if valid_loss < valid_loss_min:
                    valid_loss_min = valid_loss

                if valid_best_acc < valid_acc:
                    valid_best_acc = valid_acc

        # Otherwise increment count of epochs with no improvement
        else:
            epochs_no_improve += 1
            # Trigger early stopping
            if epochs_no_improve >= max_epochs_stop:
                print(
                    f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                )
                total_time = timer() - overall_start
                print(
                    f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                )

                # Load the best state dict
                # You can use model.load_state_dict()

                model.load_state_dict(torch.load(save_file_name))



                # Attach the optimizer
                model.optimizer = optimizer

                # Format history
                history = pd.DataFrame(
                    history,
                    columns=[
                        'train_loss', 'valid_loss', 'train_acc',
                        'valid_acc'
                    ])
                return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_save:.2f} and acc: {100 * valid_acc_save:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

# Hyperparameters
num_classes = 2
learning_rate = opt.learning_rate
#num_epochs = 10

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

saved_dir = opt.data_dir + opt.sensitive_type + '/model/' + opt.target
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

if opt.fairloss:
    save_file_name = saved_dir + '/' + opt.img_type + f'_best_model+fairloss.pt'
else:
    save_file_name = saved_dir + '/' + opt.img_type + f'_best_model.pt'
#train_on_gpu = cuda.is_available()

if opt.isTrain:
    model, history = train(model,
        criterion,
        optimizer,
        train_loader, 
        valid_loader,
        save_file_name=save_file_name,
        max_epochs_stop=opt.max_epochs_stop,
        n_epochs=opt.num_epochs,
        print_every=1)


###########################################################################
##########################  Model Testing         #########################
###########################################################################

else:
    print('Evaluating Model')
    model.load_state_dict(torch.load(save_file_name))
    model.eval()


    # Check accuracy on training to see how good our model is

    predict_output = np.zeros((test_num, 1), dtype=int)
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        i = 0
        
        
        model.eval()

        with torch.no_grad():
            for x, y, bias in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                
                predict = predictions.data.cpu().numpy()
                predict_output[i] = predict
                i+=1

                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}%"
            )

        #model.train()

    print("Checking accuracy on Test Set")
    check_accuracy(test_loader, model)
    predict_output_df = pd.DataFrame(predict_output, columns = ['predict'])


    import io

    test = pd.read_csv(test_csv)

    #test_goldout = pd.DataFrame(test['Smiling']).to_numpy()
    #abs(predict_output - test_goldout).sum()

    sensitive = test[opt.sensitive_type]
    #frames = [predict_output_df, male, test['smile']]
    frames = [predict_output_df, sensitive, test[opt.target]]
    
    result = pd.concat(frames, axis=1)

    def ypos_zpos(a,b):
        if a==1 and b==1:
            return 1
        else:
            return 0

    def ypos_zneg(a,b):
        if a==1 and b==0:
            return 1
        else:
            return 0

    result['y=1z=1'] = result.apply(lambda result: ypos_zpos(result['predict'],result[opt.sensitive_type]), axis=1)
    result['y=1z=-1'] = result.apply(lambda result: ypos_zneg(result['predict'],result[opt.sensitive_type]), axis=1)

    if result[opt.sensitive_type].value_counts().loc[0] == result[opt.sensitive_type].value_counts().loc[1]:
        pos_sensitive_num = result[opt.sensitive_type].value_counts().loc[0]

    DP = abs(result['y=1z=-1'].sum()/pos_sensitive_num - result['y=1z=1'].sum()/pos_sensitive_num)
    print('The value of DP is',DP)

    def yhat1Y1Zn1(a,b,c):
        if a==1 and b==1 and c==0:
            return 1
        else:
            return 0

    def yhat1Y1Z1(a,b,c):
        if a==1 and b==1 and c==1:
            return 1
        else:
            return 0

    def Y1Zn1(a,b):
        if a==1 and b==0:
            return 1
        else:
            return 0

    def Y1Z1(a,b):
        if a==1 and b==1:
            return 1
        else:
            return 0

    result['yhat1Y1Zn1'] = result.apply(lambda result: yhat1Y1Zn1(result['predict'],result[opt.target],result[opt.sensitive_type]), axis=1)
    result['yhat1Y1Z1'] = result.apply(lambda result: yhat1Y1Z1(result['predict'],result[opt.target],result[opt.sensitive_type]), axis=1)

    result['Y1Zn1'] = result.apply(lambda result: Y1Zn1(result[opt.target],result[opt.sensitive_type]), axis=1)
    result['Y1Z1'] = result.apply(lambda result: Y1Z1(result[opt.target],result[opt.sensitive_type]), axis=1)

    def yhat1Yn1Zn1(a,b,c):
        if a==1 and b==0 and c==0:
            return 1
        else:
            return 0

    def yhat1Yn1Z1(a,b,c):
        if a==1 and b==0 and c==1:
            return 1
        else:
            return 0

    def Yn1Zn1(a,b):
        if a==0 and b==0:
            return 1
        else:
            return 0

    def Yn1Z1(a,b):
        if a==0 and b==1:
            return 1
        else:
            return 0

    result['yhat1Yn1Zn1'] = result.apply(lambda result: yhat1Yn1Zn1(result['predict'],result[opt.target],result[opt.sensitive_type]), axis=1)
    result['yhat1Yn1Z1'] = result.apply(lambda result: yhat1Yn1Z1(result['predict'],result[opt.target],result[opt.sensitive_type]), axis=1)

    result['Yn1Zn1'] = result.apply(lambda result: Yn1Zn1(result[opt.target],result[opt.sensitive_type]), axis=1)
    result['Yn1Z1'] = result.apply(lambda result: Yn1Z1(result[opt.target],result[opt.sensitive_type]), axis=1)

    DEO = (abs(result['yhat1Y1Zn1'].sum()/result['Y1Zn1'].sum() - result['yhat1Y1Z1'].sum()/result['Y1Z1'].sum()) + 
      abs(result['yhat1Yn1Zn1'].sum()/result['Yn1Zn1'].sum() - result['yhat1Yn1Z1'].sum()/result['Yn1Z1'].sum()))

    print('The value of DEO is',DEO)