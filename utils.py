#Function files
import os
import numpy as np
import torch
import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False
from datetime import datetime
from covid_dataset import Covid19Dataset

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
date = str(datetime.now().date())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_test_dir = '/home/oury/Documents/Ram/ct_project/data/test'

def get_accuracy(outputs,covidclass):                #gives the accuracy after we got the presitions
    correct = 0
    for i in range(outputs.shape[0]):
        outputs[i] = outputs[i]>0.5
        if outputs[i] == covidclass[i]:
            correct += 1
    
    accuracy = ((correct)/int(outputs.shape[0]))
    return accuracy

def validate_model(model, val_loader, criterion, device ,covidclass):  #after each epoch - test the model on a validaion group
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

def test_evaluation(model, device, dir):
    
    test_dataset = Covid19Dataset(
    image_dir=image_test_dir)
    
    y_true = []                             #handeling the true covid class
    best_cm = 0
    best_cm_trace = 0
    best_cm_path = 0
    for i in range(len(test_dataset)):
        y_true[i] = (test_dataset[i])[1]

    ver_list = glob.glob(os.path.join(dir,"*.zip"))
    for i in ver_list:
        model.load_state_dict(torch.load(i))
        model.eval()

        y_pred = []
        for image, covidclass in test_dataset:
            image = image.to(device)
            output = model(image)
            output = output>0.5
            y_pred.append(output)

        cm = confusion_matrix(y_true,y_pred)
        cm_trace = np.trace(cm)
        if cm_trace > best_cm_trace:
            best_cm = cm
            best_cm_trace = cm_trace
            best_cm_path = ver_list[i]
    
    new_path = f"bestversion_{best_cm_path}"
    os.rename(best_cm_path,new_path)
    
    TN, FP, FN, TP = best_cm.ravel()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)         
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    y_middle = (y_min + y_max)/2
    sns.heatmap(best_cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predict Label')
    plt.text(x_max + (x_max - x_min)*0.05,y_middle, f"Precosion: {precision:.2f}\nRecall: {recall:.2f}",frontsize=12, color='black', verticalalignment='center')
    plt.show()
    plt.savefig('confusion_matrix.png')

    return best_cm

def update_plot_epoch(train_accuracy, val_accuracy, train_loss, val_loss, num_epochs, epoch,save_plot_dir):   #after each epoch - save the plot with the current time and date and update it

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
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    current_date = datetime.now().strftime('%Y-%m-%d')
    plt.savefig(os.path.join(save_plot_dir, f'training_curve_{current_date}'), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

def weight_tensor(train_dataset):                    #overcome the unbalanced dataset

# Directory containing your mask images
    class_counts = np.zeros(2)
    inverse_frequencies = np.zeros(2)
    for i in range(len(train_dataset)):
        class_counts[(train_dataset[i])[1]] += 1
    
    inverse_frequencies = 1/class_counts             # Get class frequencies

    # Normalize weights
    sum_inverse_frequencies = sum(inverse_frequencies)
    pos_weight = torch.tensor([weight / sum_inverse_frequencies for weight in inverse_frequencies])
    pos_weight = (pos_weight.view(2, 1, 1)).to(device)
    return pos_weight






