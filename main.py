from csNet import csNet
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=str, default="pretrained_model/pre-trained_models.pth")
    parser.add_argument("--optimizer_path", type=str, default="pretrained_model/optimizer.pth")
    args = parser.parse_args()
    model_path = args.model_path
    optimizer_path = args.optimizer_path

    # Get dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = csNet(device=device)

    if os.path.exists(model_path):
        model.load_models(model_path)
    if os.path.exists(optimizer_path):
        model.load_optimizers(optimizer_path)

    if args.train:
        epochs = 10
        model.train(trainloader, epochs)
        model.save_model(os.path.dirname(model_path), os.path.basename(model_path))
        model.save_optimizer(os.path.dirname(optimizer_path), os.path.basename(optimizer_path))

    # Test
    model.test_each_model_accuracy_on_certain_dataset(testloader)

    # model.test_each_model_if_average_different_dropout_is_good(testloader)

    # model.test_if_average_different_models_is_good(testloader)

    correct = 0
    total = 0
    total_samples = len(testloader)
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        predicted = model.predict(images, labels)
        correct += (predicted == labels).sum().item()
        print("Samples:[{}/{}], Accuracy:{:.4f}%".format(i+1, total_samples, correct/total * 100))

if __name__ == "__main__":
    main()