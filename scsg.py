"""
Pytorch code for Non-Convex Finite-Sum Optimization Via SCSG Methods (https://papers.nips.cc/paper/6829-non-convex-finite-sum-optimization-via-scsg-methods.pdf)
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

BATCH_SIZE = 100
LR_SCSG = 0.06

EPOCH = 500
LARGE_BATCH_NUMBER=25

# Setup DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 10))
        
    def forward(self, inputs):
        inputs = inputs.view(-1, 28 * 28)
        x = self.model(inputs)
        return F.log_softmax(x,dim=1)
    
    def partial_grad(self, inputs, targets):
        """
        Function to compute the stochastic gradient
        args : input, loss_function
        return : loss
        """
        outputs = self.forward(inputs)
        # compute the partial loss
        loss = F.nll_loss(outputs, targets)

        # compute gradient
        loss.backward()
        return loss.detach()
    
    def calculate_loss_grad(self, dataset, large_batch_num):
        """
        Function to compute the large-batch loss and the large-batch gradient
        args : dataset, loss function, number of samples to be calculated
        return : total_loss, full_grad_norm
        """

        total_loss = 0.0

        for idx, data in enumerate(dataset):
            # only calculate the sub-sampled large batch
            if idx > large_batch_num - 1:
                break
            # load input
            inputs, targets = data
            inputs, targets = torch.FloatTensor(inputs).cuda(), torch.LongTensor(targets).cuda()

            # calculate loss
            total_loss += self.partial_grad(inputs, targets)
            
        total_loss /= large_batch_num

        return total_loss

def scsg_step(net, optimizer, train_loader, test_loader, inner_iter_num):
    """
    Function to updated weights with a SCSG backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    large_batch_num = LARGE_BATCH_NUMBER
    batch_size = BATCH_SIZE

    # Compute full grad
    pre_net_full.zero_grad()
    _ = pre_net_full.calculate_loss_grad(train_loader, large_batch_num)

    running_loss = 0.0
    iter_num = 0.0

    # Run over the train_loader
    for batch_id, batch_data in enumerate(train_loader):

        if batch_id > inner_iter_num - 1:
            break

        # get the input and label
        inputs, targets = batch_data

        # wrap data and target into variable
        inputs, targets = torch.FloatTensor(inputs).cuda(), torch.LongTensor(targets).cuda()

        # compute previous stochastic gradient
        pre_net_mini.zero_grad()
        # take backward
        pre_net_mini.partial_grad(inputs, targets)

        # compute current stochastic gradient
        optimizer.zero_grad()

        outputs = net(inputs)
        current_loss = F.nll_loss(outputs, targets)
        current_loss.backward()

        # take SCSG gradient step
        for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
            p_net.grad.data += p_full.grad.data * (1.0 / large_batch_num) - p_mini.grad.data
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
        optimizer.step()

        # print statistics
        running_loss += current_loss.detach()
        iter_num += 1.0

    # calculate training loss
    train_loss = running_loss / iter_num

    # calculate test loss
    net.zero_grad()
    test_loss = net.calculate_loss_grad(test_loader, len(test_loader)/batch_size)

    return train_loss, test_loss

def sgd_step(net, optimizer, train_loader, test_loader, inner_iter_num):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    batch_size = BATCH_SIZE

    running_loss = 0.0
    iter_num = 0.0

    # Run over the train_loader
    for batch_id, batch_data in enumerate(train_loader):

        if batch_id > inner_iter_num - 1:
            break

        # get the input and label
        inputs, targets = batch_data

        # wrap data and target into variable
        inputs, targets = torch.FloatTensor(inputs).cuda(), torch.LongTensor(targets).cuda()

        # compute current stochastic gradient
        optimizer.zero_grad()

        outputs = net(inputs)
        current_loss = F.nll_loss(outputs, targets)
        current_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)

        optimizer.step()

        # print statistics
        running_loss += current_loss.detach()
        iter_num += 1.0

    # calculate training loss
    train_loss = running_loss / iter_num

    # calculate test loss
    net.zero_grad()
    test_loss = net.calculate_loss_grad(test_loader, len(test_loader)/batch_size)

    return train_loss, test_loss

net = Model().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=LR_SCSG)
# optimizer = torch.optim.Adam(net.parameters())

scsg_loss_train = []
scsg_loss_test  = []

# training
for epoch in range(EPOCH):
    inner_iter_num = np.random.geometric(1.0/(LARGE_BATCH_NUMBER + 1.0))
    # take one epoch scsg step
    cur_train_loss, cur_test_loss = scsg_step(net, optimizer, train_loader, test_loader, inner_iter_num)
    scsg_loss_train.append(cur_train_loss)
    scsg_loss_test.append(cur_test_loss)
    # print progress
    print('Epoch: ', epoch,
          '| train loss: %.8f' % cur_train_loss,
          '| test loss: %.8f' % cur_test_loss)

net = Model().cuda()
# optimizer = torch.optim.SGD(net.parameters(), lr=LR_SCSG)
optimizer = torch.optim.Adam(net.parameters())

sgd_loss_train = []
sgd_loss_test  = []

# training
for epoch in range(EPOCH):
    inner_iter_num = len(train_loader)/BATCH_SIZE
    # take one epoch scsg step
    cur_train_loss, cur_test_loss = sgd_step(net, optimizer, train_loader, test_loader, inner_iter_num)
    sgd_loss_train.append(cur_train_loss)
    sgd_loss_test.append(cur_test_loss)
    # print progress
    print('Epoch: ', epoch,
          '| train loss: %.8f' % cur_train_loss,
          '| test loss: %.8f' % cur_test_loss)

"""
Draw training loss
"""
fig, axs = plt.subplots()
x = np.arange(500)
axs.plot(x,sgd_loss_train,label='sgd_loss_train')
# axs.plot(x,sgd_loss_test,label='sgd_loss_test')
axs.plot(x,scsg_loss_train,label='scsg_loss_train')
# axs.plot(x,scsg_loss_test,label='scsg_loss_test')
        
axs.set_xlabel('iters')
axs.set_ylabel('loss')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('train_loss_scsg.pdf')

"""
Draw testing loss
"""
fig, axs = plt.subplots()
x = np.arange(500)
# axs.plot(x,sgd_loss_train,label='sgd_loss_train')
axs.plot(x,sgd_loss_test,label='sgd_loss_test')
# axs.plot(x,scsg_loss_train,label='scsg_loss_train')
axs.plot(x,scsg_loss_test,label='scsg_loss_test')
        
axs.set_xlabel('iters')
axs.set_ylabel('loss')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('test_loss_scsg.pdf')




