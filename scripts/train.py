import yaml
from models.csNet import csNet
import torchvision
import torch
import torchvision.transforms as transforms
import os
import datetime

def main(config_path):

    exp_name = "experiment_"+datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")

    # Get config file
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    seed = config["general"]["seed"]
    device = config["general"]["device"]

    dataset_shuffle = config["training"]["dataset"]["shuffle"]
    apply_transformation = config["training"]["dataset"]["apply_transformation"]
    dataset_size = config["training"]["dataset"]["dataset_size"]
    batch_size = config["training"]["dataset"]["batch_size"]

    epochs = config["training"]["epochs"]

    networks_config = config["networks"]
    optimizor_config = config["training"]["optimizor"]

    wandb_config = config["wandb"]
    if wandb_config["name"] is None:
        wandb_config["name"] = exp_name
    wandb_config["config"] = config

    paths = config["paths"]
    abs_path = os.path.dirname(config_path)
    for path_name in paths:
        try:
            paths[path_name] = os.path.join(abs_path, paths[path_name])
        except:
            pass
    paths["save_directory"] = os.path.join(paths["save_directory"], exp_name)

    # Set the seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

    # Get dataset
    if apply_transformation:
        transform = transforms.Compose(
                    [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    ])
    else:
        transform = transforms.ToTensor()        

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    index = torch.randperm(len(trainset))[:int(len(trainset)*dataset_size)]

    trainset = torch.utils.data.Subset(trainset, index)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=dataset_shuffle)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=dataset_shuffle)

    # Create directory to save models:
    if not os.path.exists(paths["save_directory"]):
        os.makedirs(paths["save_directory"])
        config_path = os.path.join(paths["save_directory"], "configs.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    else:
        raise RuntimeError("Experiments already exits")

    # Create Model
    device = torch.device(device)  
    model = csNet(paths=paths, networks_config=networks_config, device=device, seed=seed, wandb_config=wandb_config, exp_name=exp_name)

    model.train(trainloader, testloader, optimizor_config, epochs)

if __name__ == "__main__":
    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/configs.yaml",
        )
    )
    main(path_to_config_file)