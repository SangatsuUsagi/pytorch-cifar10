#! /usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.alexnet import alexnet
from models.mobilenet import mobilenetv1
from models.resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg import vgg11, vgg13, vgg16, vgg19
from models.wrn import wrn_16_10, wrn_28_10


class Params:
    def __init__(self):
        self.model = "resnet18"
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 16
        self.start_epoch = 1
        self.lr = 0.01
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 80
        self.save_model = True
        self.loss = []
        self.acc = []
        self.best_acc = 0.0
        self.resume = False
        self.workdir = "."


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    # Sets the module in training mode.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        args.loss.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader, criterion):
    # Sets the module in evaluation mode.
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_loss += (criterion(output, target)).item()
            correct += (predicted == target).sum().item()

            test_loss /= len(test_loader)

    acc = 100.0 * correct / len(test_loader.dataset)
    args.acc.append(acc)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc
        )
    )


def main():
    args = Params()

    parser = argparse.ArgumentParser(description="Train CIFAR10 with PyTorch")
    parser.add_argument(
        "--model",
        choices=[
            "alexnet",
            "resnet101",
            "resnet152",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet9",
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "wrn_16_10",
            "wrn_28_10",
        ],
        help="model for training",
    )
    parser.add_argument("--epochs", type=int, help="Epochs for training")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda")
    parser.add_argument(
        "--workdir", help="working directory to store checkpoint and weights"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    cmd_args = parser.parse_args()

    if cmd_args.model:
        args.model = cmd_args.model
    if cmd_args.epochs:
        args.epochs = cmd_args.epochs
    if cmd_args.no_cuda:
        args.no_cuda = True
    if cmd_args.resume:
        args.resume = True
    if cmd_args.workdir:
        args.workdir = cmd_args.workdir

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": os.cpu_count(), "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data/cifar10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(size=32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # Uncomment to normalize
                    # transforms.Normalize((0.4914, 0.4822, 0.4465),
                    #                      (0.2023, 0.1994, 0.2010)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data/cifar10",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Uncomment to normalize
                    # transforms.Normalize((0.4914, 0.4822, 0.4465),
                    #                      (0.2023, 0.1994, 0.2010)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = eval(args.model)().to(device)

    if args.resume:
        # Load checkpoint.
        print("* Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.workdir + "/checkpoint/ckpt_cifar10.pth")
        model.load_state_dict(checkpoint["net"])
        args.best_acc = checkpoint["acc"]
        args.start_epoch = checkpoint["epoch"] + 1

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters())

    print("Training " + args.model + " with CIFAR10 dataset")

    torch.backends.cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs + 1):

        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion)

        # Save weights/biases
        if args.save_model:
            if not os.path.isdir(args.workdir + "/weights"):
                os.mkdir(args.workdir + "/weights")
            torch.save(
                model.state_dict(),
                args.workdir
                + "/weights/cifar10-{0}_weights_{1:0>3}.pth".format(args.model, epoch),
            )

        if args.acc[-1] >= args.best_acc:
            # Save state for checkpoint
            print(
                "Saving checkpoint. (Epoch:{}, Accuracy:{})\n".format(
                    epoch, args.acc[-1]
                )
            )
            state = {
                "net": model.state_dict(),
                "acc": args.acc[-1],
                "epoch": epoch,
            }
            if not os.path.isdir(args.workdir + "/checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, args.workdir + "/checkpoint/ckpt_cifar10.pth")

            # Create symlink for the best accuracy weights
            if args.save_model:
                target_name = (
                    args.workdir
                    + "/weights/cifar10-{0}_weights_{1:0>3}.pth".format(
                        args.model, epoch
                    )
                )
                link_name = args.workdir + "/weights/cifar10-{0}_weights.pth".format(
                    args.model
                )
                if os.path.exists(link_name):
                    os.remove(link_name)
                os.symlink(os.path.basename(target_name), link_name)

            args.best_acc = args.acc[-1]

    # Checking loss and accuracy
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    y0_val = int(len(args.loss) / args.epochs)
    x0_val = range(y0_val)
    for i in range(args.epochs):
        ax[0].plot(
            x0_val,
            (args.loss)[i * y0_val: (i + 1) * y0_val],
            label="epoch #{}".format(i + 1),
        )
        ax[0].set_xlabel("epochs")
    ax[0].set_ylim(0, 2.5)
    ax[0].set_ylabel("loss")
    ax[0].set_title("CIFAR10")
    ax[0].legend()
    y1_val = len(args.acc)
    x1_val = range(y1_val)
    ax[1].plot(x1_val, args.acc)
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("acc")
    ax[1].set_title("CIFAR10")
    ax[1].legend()
    plt.show()

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":
    main()
