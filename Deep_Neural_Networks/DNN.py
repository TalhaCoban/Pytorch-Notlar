import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets


n_pts = 500
X, y = datasets.make_circles(n_pts, 2, noise = 0.1, factor = 0.2, random_state=456)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y).view(500,1)

def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1], color = "b")
    plt.scatter(X[y==1, 0], X[y==1, 1], color = "g")

scatter_plot()
plt.show()

class Model(nn.Module):
    def __init__(self,  input_size, H1, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0

torch.manual_seed(2)
model = Model(2, 4, 1)
print(list(model.parameters()))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)


epochs = 1000
losses = []
for i in range(epochs):
    y_pred = model.forward(X_data)
    loss = criterion(y_pred, y_data)
    print("epoch:", i ,"loss:", loss.item())
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

def plot_decision_boundary(X, y):
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:, 1]) + 0.25)
    #masgrid function does is it allows us to return coordinate to matrices from the input of coordinate vectors
    xx , yy = np.meshgrid(x_span,y_span)
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    pred_func = model.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx, yy, z)

x = 0
y = 0
point = torch.Tensor([x, y])
prediction = model.predict(point)
plt.plot([x], [y], marker = "o", markersize = 10, color = "red")
print("Prediction is", prediction)

plot_decision_boundary(X_data, y_data)
plt.show()

