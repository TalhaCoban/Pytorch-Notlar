import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets


n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_pts, 2, centers = centers, cluster_std = 0.40, random_state=234)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y).view(100,1)

def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1], color = "b")
    plt.scatter(X[y==1, 0], X[y==1, 1], color = "g")

class Model(nn.Module):
    def __init__(self,  input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0


torch.manual_seed(2)
model = Model(2, 1)
print(list(model.parameters()))

def get_params():
    [w, b] = model.parameters()
    w1, w2 = w.view(2)
    b = b[0]
    return (w1.item(), w2.item(), b.item())

def plot_fit(title):
    plt.title = title
    w1, w2, b1 = get_params()
    x1 = np.array([-2.0, 2.0])
    #0 = w1x1 + w2x2 + b
    x2 = -(w1 * x1 + b1) / w2
    plt.plot(x1, x2, "r")
    scatter_plot()
    plt.show()

plot_fit("inital fit")

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

epochs = 1000
losses = []
for i in range(epochs):
    y_pred = model.forward(X_data)
    loss = criterion(y_pred, y_data)
    print("epoch: ", i ,"loss: ", loss.item())

    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


point1 = torch.tensor([1.0, -1.0])
point2 = torch.tensor([-1.0, 1.0])
plt.plot(point1[0], point1[1], "ro")
plt.plot(point2[0], point2[1], "bo")
print("Red point positive probability = {}".format(model.forward(point1).item()))
print("Black point positive probability = {}".format(model.forward(point2).item()))
print("Red point is in class {}".format(model.predict(point1)))
print("Black point is in class {}".format(model.predict(point2)))
plot_fit("trained model")

plt.show()
