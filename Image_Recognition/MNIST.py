import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import requests
import PIL


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
    image = tensor.clone().detach().numpy()
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

class Classifier(nn.Module):
      
      def __init__(self, D_in, H1, H2, D_out):
            uper().__init__()
            elf.linear1 = nn.Linear(D_in, H1)
            elf.linear2 = nn.Linear(H1, H2)
            elf.linear3 = nn.Linear(H2, D_out)
      def forward(self, x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
            return x

model = Classifier(784, 125, 65, 10)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

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
            inputs = inputs.view(inputs.shape[0], -1)
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

                        val_inputs = val_inputs.view(val_inputs.shape[0], -1)
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

response = requests.get(test_link, stream=True)
img = PIL.Image.open(response.raw)
plt.imshow(img)

img = PIL.ImageOps.invert(img)
img = img.convert("1")
img = transforms(img)
plt.imshow(im_convert(img))

img = img.view(img.shape[0], -1)
output = model.forward(img)
_, pred = torch.max(output, 1)
print(pred.item())

dataiter = iter(validation_loader)
images, labels = dataiter.next()
images_ = images.view(images.shape[0], -1)
outputs = model.forward(images_)
_, preds = torch.max(outputs, 1)

fig = plt.figure(figsize = (25, 4))

for idx in np.arange(20):
      ax = fig.add_subplot(2, 10, idx + 1, xticks = [], yticks = [])
      plt.imshow(im_convert(images[idx]))
      ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx] == labels[idx] else "red"))
plt.show()


