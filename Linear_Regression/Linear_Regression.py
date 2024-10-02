import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        pred = self.linear(x)
        return pred

torch.manual_seed(2)
model = LR(input_size=1, output_size=1)
print(list(model.parameters()))


X = torch.randn(100, 1) * 10
y = X + torch.randn(100, 1) * 3


[w, b] = model.parameters()
def get_params():
    return (w[0][0].item(), b[0].item())


def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = torch.tensor([-30, 30])
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, "r")
    plt.scatter(X, y)
    plt.show()

plot_fit("initial model")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print("epoch:", i, "loss:", loss.item())
    losses.append(loss)
    """
    we must set the gradient to zero since gradients accumulate 
    """
    optimizer.zero_grad()
    """
    we take derivative of the loss function
    """
    loss.backward()
    """
    backward and having computed the gradient we update our model parameters with optimzer
    """
    optimizer.step()

plt.plot(range(epochs), losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plot_fit("Trained Model")
print(list(model.parameters()))