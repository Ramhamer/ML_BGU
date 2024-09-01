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
save_dir = '/home/oury/Documents/Ram/ct_project/plots'



def get_accuracy(outputs,covidclass):
    correct = 0
    for i in range(outputs.shape[0]):
        outputs[i] = outputs[i]>0.5
        if outputs[i] == covidclass[i]:
            correct += 1
    
    accuracy = ((correct)/int(outputs.shape[0]))
    return accuracy


def validate_model(model, val_loader, criterion, device ,covidclass):
    model.eval()
    loss_val = 0.0
    acc_val = 0
    total_val = 0

    with torch.no_grad():        # no grad = without learning rate from the validation directory
        for image, covidclass in val_loader:
            image, covidclass = image.to(device), covidclass.to(device)
            outputs = model(image)

             # Calculate accuracy and loss
            loss = criterion(outputs, covidclass)
            loss_val += loss.item() * image.size(0)
            acc_val += get_accuracy(outputs,covidclass)* image.size(0)
    
    val_loss = loss_val / len(val_loader.dataset)
    val_acc = acc_val/len(val_loader.dataset)
    return val_loss ,val_acc



def update_plot_epoch(train_accuracy, val_accuracy, train_loss, val_loss, num_epochs, epoch,save_dir):

    global train_acc_line, val_acc_line, train_loss_line, val_loss_line
    if epoch == 0:
        # Initialize the plot only once
        plt.ion()
        plt.figure(figsize=(12, 6), num='Epoch Metrics')
        

        # Subplot for accuracy
        plt.subplot(1, 2, 1)
        train_acc_line, = plt.plot([], [], 'blue', label='Training Accuracy')
        val_acc_line, = plt.plot([], [], 'orange', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.suptitle('Xception model')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
    
        # Subplot for loss
        plt.subplot(1, 2, 2)
        train_loss_line, = plt.plot([], [], 'blue', label='Training Loss')
        val_loss_line, = plt.plot([], [], 'orange', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

    # Update the data for each line
    train_acc_line.set_data(range(1, len(train_accuracy) + 1), train_accuracy)
    val_acc_line.set_data(range(1, len(val_accuracy) + 1), val_accuracy)
    
    train_loss_line.set_data(range(1, len(train_loss) + 1), train_loss)
    val_loss_line.set_data(range(1, len(val_loss) + 1), val_loss)
 
    # Update the limits for the axes
    plt.subplot(1, 2, 1)
    plt.xlim(1, num_epochs)
    
    plt.subplot(1, 2, 2)
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1.3)  # Add a bit of margin to the y-axis

    # Remove old text annotations
    for ax in plt.gcf().get_axes():
        for txt in ax.texts:
            txt.remove()

    # Add new text annotations
    plt.subplot(1, 2, 1)
    plt.text(len(train_accuracy), train_accuracy[-1], f'{len(train_accuracy)}, {train_accuracy[-1]:.2f}', color='blue', weight='bold', fontsize=10, ha='right', va='bottom')
    plt.text(len(val_accuracy), val_accuracy[-1], f'{len(val_accuracy)}, {val_accuracy[-1]:.2f}', color='orange', weight='bold', fontsize=10, ha='right', va='bottom')

    plt.subplot(1, 2, 2)
    plt.text(len(train_loss), train_loss[-1], f'{len(train_loss)}, {train_loss[-1]:.2f}', color='blue', weight='bold', fontsize=10, ha='right', va='bottom')
    plt.text(len(val_loss), val_loss[-1], f'{len(val_loss)}, {val_loss[-1]:.2f}', color='orange', weight='bold', fontsize=10, ha='right', va='bottom')

    plt.draw()
    plt.pause(0.1)

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_date = datetime.now().strftime('%Y-%m-%d')
    plt.savefig(save_dir)



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
criterion = nn.BCEWithLogitsLoss()


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
        epoch_bar.update()
        
        
        #build train dir
        # split = data_dir.split('/')
        # data_type = '_'.join(split[-2:])
        # model_name1 = f'{date}_model_name_={model_name}_epoch{epoch}_{criterion_name}_{data_type}'
        # save_dir = f'{date}_model_name_={model_name}_{criterion_name}_{data_type}'
        # os.makedirs(save_dir, exist_ok=True)


        #plot curves
        update_plot_epoch(train_accuracies, val_accuracies, train_losses,val_losses ,num_epochs,epoch,save_dir)
        # write_backup(train_accuracies, val_accuracies, train_losses,val_losses,model_name,save_dir)
        # if (epoch) % 10 == 0:
        #     # Save the model
        #     torch.save(model.state_dict(),os.path.join(save_dir,model_name1))
        #     if check_convergence(train_losses,val_losses,back_epochs,epslion):
        #         break
        




