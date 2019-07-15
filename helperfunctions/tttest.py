import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose(
                               [transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(root='./mini', transform=transform)
#print (trainset.imgs)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=6,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.ImageFolder(root='./test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=False, num_workers=1)

classes = ('0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99')

import torch.nn as nn
import torch.nn.functional as F
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5, stride=(2,2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=(2,2))
#        self.fc1 = nn.Linear(16 * 5 * 5, 2048)
        self.fc1 = nn.Linear(16 * 13 * 13, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
#        print ("before: ")
#        print (x.shape)
        x = self.pool(F.relu(self.conv1(x)))
#        print (x.shape)
        x = self.pool(F.relu(self.conv2(x)))
#        print (x.shape)
#        x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#        print (x.shape)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
#        print (labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
#        print (outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
#        if i % 20 == 19:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')

#dataiter = iter(testloader)
#images, labels = dataiter.next()
#
#imshow(torchvision.utils.make_grid(images))
#
#print (labels)
#print ('GroundTruth: ', ' '.join('%5s' % classes[labels[j] - 1] for j in range(6)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        print (labels)
        print (outputs)
#        print (predicted)
        imshow(torchvision.utils.make_grid(images))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
                                                                   100 * correct / total))
