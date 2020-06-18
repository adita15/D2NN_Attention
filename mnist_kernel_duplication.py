from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
import os
import time
import GPUtil

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1, error=0, num_duplication=20, duplicate_index=None, attention_mode=False, error_spread=False):
        super().__init__()
        self.num_classes = num_classes
        self.error = error
        self.duplicate_index = duplicate_index

        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.attention_mode = attention_mode
        self.evaluate_origin = False
        self.num_duplication = num_duplication
        self.error_spread = error_spread

        self.flat_dim = inp_size*inp_size*4
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))
        self.fc3 = nn.Linear(32, self.num_duplication, bias=False)
        self.fc4 = nn.Linear(self.num_duplication, 32, bias=False)

    def error_injection(self, x, error_rate, duplicate_index, is_origin):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        origin_shape = x.shape
        if not is_origin:
            total_dim = x[:, :self.num_duplication, :, :].flatten().shape[0]
        else:
            total_dim = x[:, :32, :, :].flatten().shape[0]
            duplicate_index = torch.arange(32).type(torch.long).to(device)
        index = torch.arange(32).type(torch.long).to(device)
        final = torch.stack((duplicate_index, index), axis=0)
        final = final.sort(dim=1)
        reverse_index = final.indices[0]
        x = x[:, duplicate_index, :, :].flatten()
        random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
        x[random_index] = 0
        x = x.reshape(origin_shape)
        x = x[:, reverse_index, :, :]

        return x

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)

        '''
            Simulate error injection
        '''

        if self.error:
            if self.attention_mode:
                x_copy = x.clone()
                if self.error_spread:
                    x = self.error_injection(x, self.error/2, self.duplicate_index, is_origin=False)
                    x_copy = self.error_injection(x_copy, self.error/2, self.duplicate_index, is_origin=False)
                else:
                    x = self.error_injection(x, self.error, self.duplicate_index, is_origin=False)

                x = (x+x_copy)/2
            else:
                x = self.error_injection(x, self.error, None, is_origin=True)

        else:
            if self.attention_mode:
                x = x.permute(0, 2, 3, 1)
                x = self.fc3(x)
                x = nn.Tanh()(x)
                x = self.fc4(x)
                x = x.permute(0, 3, 1, 2)

        x = self.pool1(x)
        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)
        out = self.fc2(out)
        return out





def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers





def main():
    # 1. load dataset and build dataloader
    model_dir = "./model/"
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

    if args.evaluate:
        evaluate(args, args.error_rate, device, test_loader)
        return

    # 2. define the model, and optimizer.
    if args.run_original:
        model = SimpleCNN(num_duplication=args.num_duplication).to(device)

    else:
        PATH = "./model/original/epoch-4.pt"
        model = SimpleCNN(num_duplication=args.num_duplication).to(device)
        model.load_state_dict(torch.load(PATH), strict=False)
        model.attention_mode = True
        added_layers = {"fc3.weight", "fc4.weight"}
        for name, param in model.named_parameters():
            if name in added_layers:
                continue
            param.requires_grad = False

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get a batch of data
            t0 = time.time()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            loss = nn.CrossEntropyLoss()(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            t1 = time.time()
            if cnt % args.log_every == 0:
                print('training timer (per {} sample)'.format(args.batch_size) + ': %.4f sec.' % (t1 - t0))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            # Validation iteration
            if cnt % args.val_every == 0:
                test_loss, test_acc = test(model, device, test_loader)
                model.train()
            cnt += 1
        scheduler.step()

        if args.run_original:
            torch.save(model.state_dict(), os.path.join(model_dir, 'original/epoch-{}.pt'.format(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, 'attention/epoch-{}.pt'.format(epoch)))


def evaluate(args, error_rate, device, test_loader):
    print("Evaluate model with error")
    PATH = "./model/attention/epoch-4.pt"

    torch.cuda.empty_cache()
    print("GPU before loading anything")
    GPUtil.showUtilization()
    print("\n")

    model = SimpleCNN(error=error_rate, num_duplication=args.num_duplication).to(device)
    model.load_state_dict(torch.load(PATH), strict=False)

    print("GPU after loading the model")
    GPUtil.showUtilization()

    print("Evaluating model without attention...")
    # evaluate the original model without attention
    model.evaluate_origin = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc without attention: ", test_acc, "\n\n\n")

    print("Evaluating model with attention...")
    # Change to evaluate the model with attention
    index = torch.arange(32).type(torch.float).to(device)
    tmp = torch.sum(model.fc3.weight, axis=0)
    final = torch.stack((tmp, index), axis=0)
    final = final.sort(dim=1, descending=True)
    all_ranks = channel_ranking(model, 'propeval')
    model.duplicate_index = all_ranks[0]
    #model.duplicate_index = final.indices[0]
    model.evaluate_origin = False
    model.attention_mode = True

    print("Evaluating model with attention with error only in original branch...")
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)

    print("Evaluating model with attention with error in both branches...")
    model.error_spread = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)



def test(model, device, test_loader):
    """Evaluate model on test dataset."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            t0 = time.time()
            data, target = data.to(device), target.to(device)
            output = model(data)
            t1 = time.time()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)


    print('testing timer (per {} sample)'.format(args.test_batch_size) + ': %.4f sec.' % (t1 - t0))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def channel_ranking(model, mode):
    all_ranks = []
    wt_1 = model.conv1.weight
    wt_2 = model.conv2.weight
    if mode == 'weightsum':
        sorted_weights_1, sorted_indices_1 = torch.sum(torch.sum(torch.sum(wt_1, dim=-1), dim=-1), dim=-1).sort(0, descending=True)
        all_ranks.append(sorted_indices_1)
    elif mode == 'propeval':
        int_sum_1 = torch.sum(wt_1, dim=1)
        int_sum_2 = torch.sum(wt_2, dim=0)
        el_prod = int_sum_1 * int_sum_2
        sorted_weights_1, sorted_indices_1 = torch.sum(torch.sum(el_prod, dim=-1), dim=-1).sort(0, descending=True)
        all_ranks.append(sorted_indices_1)

    sorted_weights_2, sorted_indices_2 = torch.sum(torch.sum(torch.sum(wt_2, dim=-1), dim=-1), dim=-1).sort(0, descending=True)
    all_ranks.append(sorted_indices_2)
    return all_ranks


def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # config for dataset
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--run_original', default=True, type=str2bool,
					help='train the original model')
    parser.add_argument('--evaluate', default=False, type=str2bool,
					help='Evaluate the model')
    parser.add_argument('--error_rate', type=float, default=0, metavar='M',
                        help='error_rate')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num_duplication', type=int, default=20, metavar='N',
                        help='number of duplication layers (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before evaluating model')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


if __name__ == '__main__':
    args, device = parse_args()
    main()
