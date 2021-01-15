#! /usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152


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

    parser = argparse.ArgumentParser(description="Train FashionMNIST with PyTorch")
    parser.add_argument(
        "--model",
        choices=[
            "resnet9",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
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
        datasets.FashionMNIST(
            "./data/fashion_mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Uncomment for extending to 3ch
                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data/fashion_mnist",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Uncomment for extending to 3ch
                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    model = eval(args.model)(in_size=28, num_classes=10, grayscale=True).to(device)

    if args.resume:
        # Load checkpoint.
        print("* Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.workdir + "/checkpoint/ckpt_fashion_mnist.pth")
        model.load_state_dict(checkpoint["net"])
        args.best_acc = checkpoint["acc"]
        args.start_epoch = checkpoint["epoch"] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Training " + args.model + " with FashionMNIST dataset")

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
                + "/weights/fashion_mnist-{0}_weights_{1:0>3}.pth".format(
                    args.model, epoch
                ),
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
            torch.save(state, args.workdir + "/checkpoint/ckpt_fashion_mnist.pth")

            # Create symlink for the best accuracy weights
            if args.save_model:
                target_name = (
                    args.workdir
                    + "/weights/fashion_mnist-{0}_weights_{1:0>3}.pth".format(
                        args.model, epoch
                    )
                )
                link_name = (
                    args.workdir
                    + "/weights/fashion_mnist-{0}_weights.pth".format(args.model)
                )
                if os.path.exists(link_name):
                    os.remove(link_name)
                os.symlink(os.path.basename(target_name), link_name)

            args.best_acc = args.acc[-1]

    # checking loss and accuracy
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    y0_val = int(len(args.loss) / args.epochs)
    x0_val = range(y0_val)
    for i in range(args.epochs):
        ax[0].plot(
            x0_val,
            (args.loss)[i * y0_val : (i + 1) * y0_val],
            label="epoch #{}".format(i + 1),
        )
        ax[0].set_xlabel("epochs")
    ax[0].set_ylim(0, 2.5)
    ax[0].set_ylabel("loss")
    ax[0].set_title("FashionMNIST")
    ax[0].legend()
    y1_val = len(args.acc)
    x1_val = range(y1_val)
    ax[1].plot(x1_val, args.acc)
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("acc")
    ax[1].set_title("FashionMNIST")
    ax[1].legend()
    plt.show()

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":
    main()
