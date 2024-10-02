import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import requests
import PIL
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([  
                                       transforms.Resize((32, 32)), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(10),
                                       transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])

transforms = transforms.Compose([
                                transforms.Resize((32, 32)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

training_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms_train)
validation_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * (np.array(0.5,) + np.array(0.5,))
    image = image.clip(0, 1) 
    image = np.squeeze(image)
    return image

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize = (20, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1)
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])
fig.tight_layout()
plt.show()


class LeNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding = True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding = True)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding = True)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding = True)
        self.dropout2 = nn.Dropout(0.3)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding = True)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding = True)
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(4 * 4 * 128, 500)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout3(x)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

model = LeNet().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 25
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):

    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    for inputs, labels in training_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item()
    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
          
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model.forward(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += torch.sum(val_preds == val_labels.data)
                val_running_loss += val_loss.item()

        epoch_loss = running_loss / len(training_loader.dataset)
        epoch_acc = running_corrects.float() / len(training_dataset)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        epoch_val_loss = val_running_loss / len(validation_loader.dataset)
        epoch_val_acc = val_running_corrects.float() / len(validation_dataset)
        val_running_loss_history.append(epoch_val_loss)
        val_running_corrects_history.append(epoch_val_acc)

        print("loss: {:.4f} - acc_: {:.4f} ---- val_loss: {:.4f} - val_acc_: {:.4f}". format(epoch_loss, epoch_acc.item(), epoch_val_loss, epoch_val_acc.item()))
        print("Epoch: {}/{} [====================>]  done!\n".format(e + 1, epochs))

plt.plot(running_loss_history, label="Training loss")
plt.plot(val_running_loss_history, label="validation loss")
plt.legend()
plt.show()

plt.plot(running_corrects_history, label="Training accuracy")
plt.plot(val_running_corrects_history, label="validaiton accuracy")
plt.legend()
plt.show()

test_images = []
for i in range(7):
    img_path = os.path.join("test_images/{}.jpg".format(str(i+1)))
    test_images.append(PIL.Image.open(img_path))

fig, ax = plt.subplots(1, len(test_images), figsize = (12,4))
fig.tight_layout()

i = 0
for image in test_images:
    img = transforms(image)
    img = img.to(device)
    output = model.forward(img.reshape(1, 3, 32, 32))
    _, pred = torch.max(output, 1)
    ax[i].imshow(image)
    ax[i].set_title("{}(cat)".format(classes[pred.item()]), color = ("green" if str(pred.item()) == "3" else "red"))
    print("The prediction is", classes[pred.item()])
    i += 1
plt.show()


test_link = "https://s3-prod.chicagobusiness.com/A321XLR_CFM_UAL_V07.jpg"

response = requests.get(test_link, stream=True)
img = PIL.Image.open(response.raw)
plt.imshow(img)
plt.show()


img = transforms(img)
#plt.imshow(im_convert(img))

img = img.to(device)
output = model.forward(img.reshape(1, 3, 32, 32))
_, pred = torch.max(output, 1)
print("prediction:",classes[pred.item()])

dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)

outputs = model.forward(images)
_, preds = torch.max(outputs, 1)

fig = plt.figure(figsize = (25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks = [], yticks = [])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx] == labels[idx] else "red"))
fig.tight_layout()
plt.show()

sorgu = input("model kaydedilsin mi? [y] [n]")
if sorgu == "y":   
    torch.save(model.state_dict(),"model.h5")

