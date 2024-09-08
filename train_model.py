# Define the Training Loop
import os
from tqdm.auto import tqdm
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from covid_dataset import Covid19Dataset
from my_Xception import Xception 
from utils import *
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
date = str(datetime.now().date())

image_train_dir = '/home/oury/Documents/Ram/ct_project/data/train'
train_transform = True
image_val_dir = '/home/oury/Documents/Ram/ct_project/data/validation'
batch_size = 16
num_epochs = 100
save_plot_dir = '/home/oury/Documents/Ram/ct_project/results/plots'
save_dir = '/home/oury/Documents/Ram/ct_project/results'

#Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Covid19Dataset(             
    image_dir=image_train_dir,
    transform = train_transform)

val_dataset = Covid19Dataset(
    image_dir=image_val_dir)

#DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model, loss function, and optimizer
model = Xception()
optimizer = optim.Adam(model.parameters())
pos_weight = weight_tensor(train_dataset)
criterion = nn.BCEWithLogitsLoss(pos_weight)

# Training loop
num_iter = len(train_dataset)/batch_size        #how many batches there is in 1 epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
model.to(device)

with tqdm(total=num_epochs, desc="Training Progress",ncols=150, unit='epoch') as epoch_bar:
    for epoch in range(num_epochs):
        with tqdm(total=num_iter, desc="batch Progress",ncols=100  , unit='iter') as iter_bar:
            batch_accuracy =[]
            batch_loss = []
            model.train()
            acc_train= 0 
            running_loss = 0.0
            for image, covidclass in train_loader:   
                image, covidclass = image.to(device), covidclass.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(image)
                loss = criterion(outputs, covidclass)
                outputs = F.sigmoid(outputs)     #BCEWithLogitsLoss request without softmax

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Calculate accuracy and lose of iter
                item_accuracy = get_accuracy(outputs,covidclass)
                batch_accuracy.append(item_accuracy)
                batch_loss.append(loss.item())
                
                # Calculate accuracy and lose of epoch
                running_loss += loss.item() * image.size(0) # multiply batch loss by the size batch
                acc_train += item_accuracy * image.size(0)  # same as the loss
                iter_bar.update(1) #update

        epoch_loss = running_loss / len(train_loader.dataset) # divide the loss by all the data set
        epoch_acc = acc_train/len(train_loader.dataset)       # same as loss
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
    
        # Validation step
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device ,covidclass)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # print the processes
        sys.stdout.flush()
        print("\naccuracy train:" ,epoch_acc , "loss train:" ,epoch_loss , "\n" "accuracy val:" , val_accuracy," loss val:",val_loss)
        epoch_bar.update(1)    #update
        
        
        #build train dir
        # split = data_dir.split('/')
        # data_type = '_'.join(split[-2:])
        model_name = f'{date}_epoch{epoch}'
        # save_dir = f'{date}_model_name_={model_name}_{criterion_name}_{data_type}'
        # os.makedirs(save_dir, exist_ok=True)

        #plot curves
        update_plot_epoch(train_accuracies, val_accuracies, train_losses,val_losses ,num_epochs,epoch,save_dir)
        # write_backup(train_accuracies, val_accuracies, train_losses,val_losses,model_name,save_dir)
        
        #Save the model
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(),os.path.join(save_dir,model_name))
            # if check_convergence(train_losses,val_losses,back_epochs,epslion):
            #     break
        

'''
        #לשמור פלוטים והרצות
        #לכתוב הסברים על כל פונקציה
        בסוף: להריץ את הקובץ טסט ולראות שהוא לא לומד ממנו ולא שומר את המשקולות שלו
        לבדוק ארכיטקטורה אחרי שהרצנו כמה אפוקים
        לבחור את הגרסא הכי טובה של הארכיטקטורה (אחרי 20 אפוקים ואז ברח לאובר פיט - אז לעצור שם ולקחת את המשוקולת ובלה בלה בלה..
        עיצובים אחרונים (קונפיוזן מטריקס, להסתכל כל מה שניר רצה ולראות איפה מכניסים את זה)
        )

'''



