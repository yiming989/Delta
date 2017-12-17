# A simple implementation of BackPropagation Neural Network in PyTorch framework
# Attention: the BPNN's structure is stable in the code.

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data


class BPNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(BPNeuralNetwork, self).__init__()

        self.h1 = torch.nn.Linear(2, 10)
        self.h2 = torch.nn.Linear(10, 5)
        self.o = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.o(x)

        return x


if __name__ == "__main__":
    model = BPNeuralNetwork()
    leaning_rate = 1e-2
    max_epoches = 500

    train_x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_y = torch.FloatTensor([[0], [1], [1], [0]])
    test_x = torch.FloatTensor([[0, 1]])

    torch_dataset = Data.TensorDataset(data_tensor=train_x, target_tensor=train_y)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=leaning_rate)
    loss_function = torch.nn.MSELoss()

    for epoch in range(max_epoches):
        for i in range(len(train_x)):
            instance = Variable(train_x[i])
            label = Variable(train_y[i])

            output = model(instance)
            loss = loss_function(output, label)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_x = Variable(test_x)
    print(model.forward(test_x))