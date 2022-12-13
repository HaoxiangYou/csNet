import torch
import torchvision
import datetime
import argparse
import yaml
import os
from torchvision import transforms
from models.csNet import csNet

def main(config_path):

    exp_name = "experiment_"+datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")

    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained_model/Hien_pretrained_models/experiment_11-Dec-2022_00_12_05/checkpoints/model_epoch50.pth")
    args = parser.parse_args()

    # Get config file
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    seed = config["general"]["seed"]
    device = config["general"]["device"]

    batch_size = config["testing"]["dataset"]["batch_size"]
    shuffle = config["testing"]["dataset"]["shuffle"]
    apply_transformation = config["testing"]["dataset"]["apply_transformation"]
    dataset_size = config["testing"]["dataset"]["dataset_size"]

    ensemble_method = config["testing"]["ensemble"]["method"]

    networks_config = config["networks"]
    wandb_config = config["wandb"]

    # Create dataset
    if apply_transformation:
        transform = transforms.Compose(
                    [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    ])
    else:
        transform = transforms.ToTensor()  
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    
    index = torch.randperm(len(testset))[:int(len(testset)*dataset_size)]

    testset = torch.utils.data.Subset(testset, index)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=shuffle)

    device = torch.device(device)

    paths = config["paths"]

    exp_name = "experiment_"+datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    paths["save_directory"] = os.path.join(os.path.dirname(config_path), os.path.join(paths["save_directory"], exp_name))
    paths["model_path"] = args.model_path

    model = csNet(networks_config, paths=paths, wandb_config=wandb_config, seed=seed, device=device, exp_name=exp_name)

    model.test_each_model_accuracy_on_certain_dataset(testloader)

    # model.test_each_model_if_average_different_dropout_is_good(testloader)

    model.test_if_average_different_models_is_good(testloader)

    # correct = 0
    # total = 0
    # total_samples = len(testloader)
    # for i, (images, labels) in enumerate(testloader):
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     total += labels.size(0)
    #     predicted = model.predict(images, labels)
    #     correct += (predicted == labels).sum().item()
    #     print("Samples:[{}/{}], Accuracy:{:.4f}%".format(i+1, total_samples, correct/total * 100))

if __name__ == "__main__":
    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/configs.yaml",
        )
    )
    main(path_to_config_file)