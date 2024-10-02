import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
                                transforms.Resize((28, 28)), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
training_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms)
validation_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * (np.array(0.5,) + np.array(0.5,))
    image = image.clip(0, 1) 
    image = np.squeeze(image)
    return image

dataiter = iter(training_loader)
images, labels = dataiter.next()
fig = plt.figure(figsize = (25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1)
    plt.imshow(im_convert(images[idx]))
    ax.set_title(labels[idx].item())
plt.show()

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

model = LeNet().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

epochs = 15
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

epochs = 15
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

test_link = "https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg"

import requests
import PIL
response = requests.get(test_link, stream=True)
img = PIL.Image.open(response.raw)
plt.imshow(img)
plt.show()

img = PIL.ImageOps.invert(img)
img = img.convert("1")
img = transforms(img)
plt.imshow(im_convert(img))
plt.show()

img = img.to(device)
img = img[0].unsqueeze(0).unsqueeze(0)
output = model.forward(img)
_, pred = torch.max(output, 1)
print(pred.item())

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
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx] == labels[idx] else "red"))
