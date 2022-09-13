import torch
import torchvision.models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                # transforms.Normalize((0.1307,), (0.0381,))
                                ])
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True,transform=transform)
train_loader = DataLoader(train_dataset,shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True,transform=transform)
test_loader = DataLoader(test_dataset,shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
        self.convpre = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.net=net=torchvision.models.resnet18(pretrained=True)
        self.fc1=torch.nn.Linear(1000,10)

    def forward(self, x):
        # batch_size = x.size(0)
        # x = F.relu(self.pooling(self.conv1(x)))
        # x = F.relu(self.pooling(self.conv2(x)))
        # x = x.view(batch_size, -1)
        # x = self.fc(x)
        x=self.fc1(self.net(self.convpre(x)))
        return x


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 2 == 0:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0
        if batch_idx==30:
            break


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        acc=0.
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            pred=torch.argmax(outputs,dim=1)
            acc+=(pred.eq(labels)).sum().item()
            # _, predicted = torch.max(outputs.data, dim=1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
    # print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    print(acc)

if __name__ =='__main__':
    for epoch in range(1):
        train(epoch)
        test()
