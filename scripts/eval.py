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
    parser.add_argument("--model_path", type=str)
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
    if wandb_config["name"] is None:
        wandb_config["name"] = exp_name
    wandb_config["config"] = config

    # Set the seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

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
    abs_path = os.path.dirname(config_path)
    for path_name in paths:
        try:
            paths[path_name] = os.path.join(abs_path, paths[path_name])
        except:
            pass
    paths["save_directory"] = os.path.join(paths["save_directory"], exp_name)

    if args.model_path:
        paths["model_path"] = args.model_path

    model = csNet(networks_config, paths=paths, wandb_config=wandb_config, seed=seed, device=device, exp_name=exp_name)

    model.eval_each_model_accuracy(testloader)

    model.eval(testloader, method=ensemble_method)

if __name__ == "__main__":
    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/configs.yaml",
        )
    )
    main(path_to_config_file)