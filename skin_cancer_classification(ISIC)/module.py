import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models

#####################################################################
############################   BACKBONE     #########################
#####################################################################
# feature_extract is a boolean that defines if we are finetuning or feature extracting.
# If feature_extract = False, the model is finetuned and all model parameters are updated.
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

########################################################################
############################### Descriminator ##########################
########################################################################

# Without Orthogonality Constraint
class Discriminator_withoutori(nn.Module):
    def __init__(self):
        super(Discriminator_withoutori, self).__init__()
        self.fc1_1 = nn.Linear(50176, 1024) 
        self.relu = nn.ReLU()
        self.fc2_1 = nn.Linear(50176, 1024)
        self.fc1_2 = nn.Linear(1024, 128)
        self.fc2_2 = nn.Linear(1024, 128)
        self.fc1_3 = nn.Linear(128, 1)
        self.fc2_3 = nn.Linear(128, 1)

    #dimension z1 : (batch_size, 1024), dimension z2: (batch_size, 1024)
    def forward(self, x):
        x = x.view(-1, 50176)
        #discriminator branch
        x1 = self.relu(self.fc1_1(x))
        x1 = self.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)
        
        #fairness prediction branch
        x2 = self.relu(self.fc2_1(x))
        x2 = self.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)

        return F.sigmoid(x1), x2

# With Orthogonality Constraint
class Discriminator_withori(nn.Module):
    def __init__(self):
        super(Discriminator_withori, self).__init__()
        self.fc1_1 = nn.Linear(50176, 1024) 
        self.relu = nn.ReLU()
        self.fc2_1 = nn.Linear(50176, 1024)
        self.fc1_2 = nn.Linear(1024, 128)
        self.fc2_2 = nn.Linear(1024, 128)
        self.fc1_3 = nn.Linear(128, 1)
        self.fc2_3 = nn.Linear(128, 1)

    #dimension z1 : (batch_size, 1024), dimension z2: (batch_size, 1024)
    def forward(self, x, z1, z2):
        x = x.view(-1, 50176)
        #discriminator branch
        x1 = self.relu(self.fc1_1(x))
        x1 = x1 + z1
        x1 = self.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)
        
        #fairness prediction branch
        x2 = self.relu(self.fc2_1(x))
        x2 = x2 + z2
        x2 = self.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)

        return F.sigmoid(x1), x2