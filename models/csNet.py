import os
import json
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage, ToTensor
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import wandb

# Defining the model

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class basic_cnn_t(nn.Module):
    def __init__(self, c1=96, c2=192, d1=0.2, d2=0.5):
        super().__init__()

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            nn.Dropout(d1),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(d2),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(d2),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

    def forward(self, x):
        return self.m(x)

class basic_transform(nn.Module):

    def __init__(self, policy = None, device=torch.device("cpu")) -> None:
        super().__init__()

        if policy is None:
            policy = basic_transform.generate_random_policy()        
        self.policy = policy
    
        self.device = device

    @staticmethod
    def generate_random_policy():
        policies = [
            ("ShearX", np.random.random()*0.6 - 0.3),
            ("ShearY", np.random.random()*0.6 - 0.3),
            ("Rotate", np.random.random()*60 - 30),
            ("Brightness", np.random.random()*1.8 - 0.9),
            ("Color", np.random.random()*1.8 - 0.9),
            ("Contrast", np.random.random()*1.8 - 0.9),
            ("Sharpness", np.random.random()*1.8 - 0.9),
            ("Solarize", np.random.random()),
            ("Invert", True),
            ]
        policy = [policies[i] for i in  np.random.choice(len(policies), np.random.choice(len(policies)), replace=False)]
        
        return policy

    @staticmethod
    def generate_CIFAR10_augmentation_policy():
        """
        This is random policy generation based on pytorch autoaugmentation for CIFAR10
        """
        def _augmentation_space(num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
            return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
            }
            
        def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:

            policy_id = int(torch.randint(transform_num, (1,)).item())
            probs = torch.rand((2,))
            signs = torch.randint(2, (2,))

            return policy_id, probs, signs

        policies = [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]

        height = 32
        width = 32

        policy = []

        transform_id, probs, signs = get_params(len(policies))

        op_meta = _augmentation_space(10, (height, width))
        for i, (op_name, p, magnitude_id) in enumerate(policies[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                
                policy.append((op_name, magnitude))

        return policy

    def forward(self, imgs):
        images = torch.clone(imgs)
        for i in range(imgs.shape[0]):
            for op_name, magnitude in self.policy:
                try:
                    images[i] = _apply_op(images[i], op_name, magnitude, InterpolationMode.NEAREST, None).to(self.device)
                except:
                    images[i] = ToTensor()(_apply_op(ToPILImage()(images[i]), op_name, magnitude, InterpolationMode.NEAREST, None)).to(self.device)
        return images

class csNet(nn.Module):
    
    def __init__(self, networks_config, paths=None, wandb_config=None, device=torch.device("cpu"), seed=0, exp_name="csNet") -> None:
        super().__init__()

        self.exp_name = exp_name

        self.classes = ('plane', 'car', 'bird', 'cat', 
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.device = device

        self.seed = seed 
        self.set_seed(self.seed)

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.nets = []
        self.lr_schedulers = []
        self.optimizers = []
        self.transforms_policy = []
        self.transforms = []
        self.enable_wandb = False

        if wandb_config is not None and wandb_config["enable"]:
            wandb.init(
                project=wandb_config["project_name"],
                entity=wandb_config["entity"],
                name=wandb_config["name"]
            )
            self.enable_wandb = True

        if paths is not None:
            self.save_dir = paths["save_directory"]

        self.initialize_models(networks_config=networks_config, paths=paths)

        print("Finish initilization, number of models: ", self.num_of_models)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.seed()
        np.random.seed(seed)
        torch.backends.cudnn.deterministic=True

    def save_model(self, dir=None, suffix=None, save_optimizer=False):

        dir = os.path.join(self.save_dir, dir)

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        if suffix:
            model_path = os.path.join(dir, "model_{}.pth".format(suffix))
        else:
            model_path = os.path.join(dir, "model.pth")

        nets = []
        transforms = []

        for net in self.nets:
            nets.append(net.state_dict())

        for transform in self.transforms:
            transforms.append(transform.policy)

        model = {"nets":nets,
                "transforms":transforms}

        torch.save(model, model_path)

        print("Saving models to:" + model_path)

        if save_optimizer:
            
            if suffix:
                optimizer_path = os.path.join(dir, "optimizer_{}.pth".format(suffix))
            else:
                optimizer_path = os.path.join(dir, "optimizer.pth")
            
            optimizers = []
            lr_schedulers = []

            for optimizer in self.optimizers:
                optimizers.append(optimizer.state_dict())

            for lr_scheduler in self.lr_schedulers:
                lr_schedulers.append(lr_scheduler.state_dict())

            optimizers_info = {"optimizers": optimizers, 
                            "lr_schedulers": lr_schedulers}
        
            torch.save(optimizers_info, optimizer_path)

            print("Saving optimizers to:" + optimizer_path)

    def initialize_models(self, networks_config, paths=None):
        try:
            models = torch.load(paths["model_path"])
            nets = models["nets"]
            transforms = models["transforms"]            
            self.num_of_models = len(nets)

            for i in range(self.num_of_models):
                self.nets.append(basic_cnn_t(c1=networks_config["c1"], c2=networks_config["c2"],
                                            d1=networks_config["d1"], d2=networks_config["d2"]))
                self.nets[i].load_state_dict(nets[i])

                self.transforms_policy.append(transforms[i])

            print("Initialize models and transforms from:" + paths["model_path"])
        except:
            try:
                with open(paths["transforms_path"], "r") as f:
                    self.transforms_policy = json.load(f)

                self.num_of_models = len(self.transforms_policy)

                for i in range(self.num_of_models):
                    self.nets.append(basic_cnn_t(c1=networks_config["c1"], c2=networks_config["c2"],
                                            d1=networks_config["d1"], d2=networks_config["d2"]))
                print("Initialize transforms from:" + paths["transforms_path"])
            except:
                self.num_of_models = networks_config["number_of_models"]

                if networks_config["transforms"]["include_identity"]:
                    self.transforms_policy.append([("Identity", True)])
                
                for _ in range(int(networks_config["transforms"]["include_identity"]), self.num_of_models):
                    if networks_config["transforms"]["policy"] == "CIFR10":
                        policy = basic_transform().generate_CIFAR10_augmentation_policy()
                        while len(policy) < 2:
                            policy = basic_transform().generate_CIFAR10_augmentation_policy()
                    else:
                        policy = basic_transform().generate_random_policy()
                    self.transforms_policy.append(policy)

                for i in range(self.num_of_models):
                    self.nets.append(basic_cnn_t(c1=networks_config["c1"], c2=networks_config["c2"],
                                            d1=networks_config["d1"], d2=networks_config["d2"]))

        for transforms_policy in self.transforms_policy:
            self.transforms.append(basic_transform(policy=transforms_policy, device=self.device))

        for net in self.nets:            
            net.to(self.device)

    def initialize_optimizers(self, optimizer_config, paths=None):

        for i in range(self.num_of_models):
            self.optimizers.append(optim.SGD(self.nets[i].parameters(), 
                optimizer_config["lr"], momentum=optimizer_config["momentum"], 
                weight_decay=optimizer_config["weight_decay"], nesterov=optimizer_config["nesterov"]))
            self.lr_schedulers.append(optim.lr_scheduler.ExponentialLR(optimizer=self.optimizers[i], 
                gamma=optimizer_config["lr_decay"]))
        try:
            optimizers_info = torch.load(paths["optimizer_path"])
            optimizers = optimizers_info["optimizers"]
            lr_schedulers = optimizers_info["optimizers"]
            for i in range(self.num_of_models):
                self.optimizers[i].load_state_dict(optimizers[i])
                self.lr_schedulers[i].load_state_dict(lr_schedulers[i])
            print("Loading optimizer from:" + paths["optimizer_path"])
        except:
            pass

        self.lr_decay_end_epoch = optimizer_config["end_epoch"]

    def train(self, train_loader, test_loader, optimizer_config, epochs):

        self.initialize_optimizers(optimizer_config)

        with open(os.path.join(self.save_dir, "tranforms.json"), "w") as f:
            json.dump(self.transforms_policy, f, indent=4)

        total_step = len(train_loader)

        for epoch in range(epochs):

            for j in range(self.num_of_models):
                self.nets[j].train()

                total = 0
                correct = 0
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    total += labels.size(0)
                    outputs = self.nets[j](self.transforms[j](deepcopy(images)))
                    loss = self.criterion(outputs, labels)
                    self.optimizers[j].zero_grad()
                    loss.backward()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    self.optimizers[j].step()
                    if (i+1) % (len(train_loader)//3) == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Model [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch+1, epochs, i+1, total_step, j+1, self.num_of_models, loss.item(), correct/total * 100))
                        if self.enable_wandb:
                            wandb.log({"model_"+str(j+1): {"training":{"epoch": epoch+1, "step":i+1 + epoch * len(train_loader), "loss":loss.item(),"accuracy": correct/total * 100}}})

            if epoch +1 <= self.lr_decay_end_epoch:
                for j in range(self.num_of_models):
                    self.lr_schedulers[j].step()

            if (epoch + 1) % 10 == 0:
                for j in range(self.num_of_models):
                    self.nets[j].eval()
                    total = 0
                    correct = 0
                    with torch.no_grad():
                        for i, (images, labels) in enumerate(test_loader):
                            images = images.to(self.device)
                            labels = labels.to(self.device)
                            total += labels.size(0)
                            outputs = self.nets[j](self.transforms[j](images))
                            _, predicted = torch.max(outputs.data, 1)
                            correct += (predicted == labels).sum().item()
                    print("Epoch [{}/{}], Model [{}/{}], Test accuracy:{:4f}%".format(epoch+1, epochs, j+1, self.num_of_models, correct/total * 100))
                    if self.enable_wandb:
                        wandb.log({"model_"+str(j+1): {"testing":{"epoch": epoch+1, "accuracy": correct/total * 100}}})

                self.save_model(dir="checkpoints", suffix="epoch{}".format(epoch+1), save_optimizer=True)
        
        self.save_model(save_optimizer=True)

    def adversarially_attack(self, images, labels, adversarial_type="single model", noise_mag=0.01, num_of_models=None, num_of_different_result=16):
        
        if num_of_models is None:
            num_of_models = self.num_of_models

        noises = torch.zeros((num_of_models,)+images.size(),device=self.device)

        for i in range(num_of_models):
            try:
                self.nets[i].eval()
                images_copy = deepcopy(images)
                images_copy.requires_grad = True
                outputs = self.nets[i](self.transforms[i](images_copy))
                loss= self.criterion(outputs, labels)
                self.nets[i].zero_grad()
                loss.backward()
                noises[i] = images_copy.grad.data.clone()
            except:
                pass

        if adversarial_type == "single model":
            noises = noises[0]
        elif adversarial_type == "average":
            noises = noises.mean(dim=0)
        elif adversarial_type == "weighted sum":
            _, weights = self.predicted_by_weighted_sum(images, num_of_models, num_of_different_result)
            noises = torch.sum(noises * (torch.transpose(weights, 0, 1)[:,:,None,None,None]), dim=0)
        else:
            raise ValueError("Adversarially methed only support (single model, average, weighted_sum)")

        noises = noise_mag * torch.sign(noises)

        return noises

    def eval_model_with_adversarially_attack(self, test_loader, method="single model", adversarial_type="weighted sum", num_of_models=None, num_of_different_result=16, noise_mag=0.01):
        if num_of_models is None:
            num_of_models = self.num_of_models
        
        correct = 0
        total = 0
        num_of_samples = len(test_loader)

        for j, (images, labels) in enumerate(test_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            noises = self.adversarially_attack(images, labels, adversarial_type=adversarial_type, noise_mag=noise_mag)
            images += noises

            if method == "single model":
                outputs = self.nets[0](self.transforms[0](images))
                _, predicted = torch.max(outputs.data, 1)
            elif method == "average":
                predicted = self.predict_by_simple_average(images, num_of_models)
            elif method == "weighted sum":
                predicted, _ = self.predicted_by_weighted_sum(images, num_of_models, num_of_different_result)
            else:
                raise ValueError("Only support (single model, average, weighted_sum) for adverarially attack evaluation")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("Samples:[{}/{}], Accuracy:{:.4f}%".format(j+1, num_of_samples, correct/total * 100))
                
            if self.enable_wandb:
                wandb.log({"Testing":{"num of samples":total, "adversarially attack " + method :{"accuracy": correct/total}}})
        
    def eval_each_model_accuracy(self, test_loader):
        for i in range(self.num_of_models):
            self.nets[i].eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for j, (images, labels) in enumerate(test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.nets[i](self.transforms[i](images))
                    val_loss += self.criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if self.enable_wandb:
                        wandb.log({"Testing":{"num of samples":total, "single_model_accuracy":{i:{"accuracy": correct/total}}}})
            print("Model:[{}/{}], Accuracy:{:.4f}%".format(i+1, self.num_of_models, correct/total * 100))

    def eval(self, test_loader, method="average", num_of_models=None, num_of_different_result=16):
        if num_of_models is None:
            num_of_models = self.num_of_models
        
        with torch.no_grad():
            correct = 0
            total = 0
            num_of_samples = len(test_loader)
            for j, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if method == "average":
                    predicted = self.predict_by_simple_average(images, num_of_models)
                elif method == "Kalman filter":
                    predicted = self.predict_by_kalman_filter(images, num_of_models, num_of_different_result)
                elif method == "weighted sum":
                    predicted, _ = self.predicted_by_weighted_sum(images, num_of_models, num_of_different_result)
                elif method == "majority voting":
                    predicted = self.predicted_by_majority_voting(images, num_of_models)
                else:
                    raise ValueError("Method should be in (average, Kalman filter, weighted sum, majority voting)")

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                print("Samples:[{}/{}], Accuracy:{:.4f}%".format(j+1, num_of_samples, correct/total * 100))
                
                if self.enable_wandb:
                    wandb.log({"Testing":{"num of samples":total, method:{"accuracy": correct/total}}})

    def predict_by_simple_average(self, images, num_of_models):
        for i in range(num_of_models):
            self.nets[i].eval()
        
        outputs = self.softmax(self.nets[0](self.transforms[0](deepcopy(images))))
        
        for i in range(1, num_of_models):
            outputs += self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))
        
        outputs /= num_of_models
        _, predicted = torch.max(outputs.data, 1)
        
        return predicted

    def predict_by_kalman_filter(self, images, num_of_models, num_of_different_result):

        def kalman_updates(mean_1, mean_2, cov_1, cov_2):
            K_1 = cov_2 @ torch.linalg.pinv(cov_1 + cov_2)
            K_2 = torch.eye(K_1.shape[0], device=self.device) - K_1
            mean = K_1 @ mean_1 +  K_2 @ mean_2
            cov = K_1 @ cov_1 @ K_1.T + K_2 @ cov_2 @ K_2.T
            return mean, cov  
        
        num_of_class = len(self.classes)

        # obtained predictions with multiple value, the dim are arange as  num_of_images, num_of_model, num_of_different_try, num_of_class)
        stochastic_predictions = self.obtained_stochastic_results_by_dropout(images, num_of_models, num_of_different_result)

        means = torch.mean(stochastic_predictions, dim=2)
        
        covs = torch.zeros(images.shape[0], num_of_models, num_of_class, num_of_class, device=self.device)

        for i in range(images.shape[0]):
            for j in range(num_of_models):
                covs[i][j] = torch.cov(stochastic_predictions[i][j].T)

        outputs = torch.zeros(images.shape[0], num_of_class, device=self.device)

        for i in range(images.shape[0]):
            mean = means[i][0]
            cov = covs[i][0]

            for j in range(1, num_of_models):
                mean, cov = kalman_updates(mean, means[i][j], cov, covs[i][j])
            
            outputs[i] = mean

        _, predicted = torch.max(outputs.data, 1)

        return predicted

    def predicted_by_weighted_sum(self, images, num_of_models, num_of_different_result):

        num_of_class = len(self.classes)

        # obtained predictions with multiple value, the dim are arange as  num_of_images, num_of_model, num_of_different_try, num_of_class)
        stochastic_predictions = self.obtained_stochastic_results_by_dropout(images, num_of_models, num_of_different_result)

        means = torch.mean(stochastic_predictions, dim=2)
        
        vars = torch.var(stochastic_predictions, dim=(2,3))

        outputs = torch.zeros(images.shape[0], num_of_class, device=self.device)

        weights = torch.zeros(images.shape[0], num_of_models, device=self.device)

        for i in range(images.shape[0]):
            mean = means[i][0]
            var = vars[i][0]

            weights[i][0] = 1

            for j in range(1, num_of_models):
                k = vars[i][j] / (vars[i][j] + var)
                mean = k * mean + (1-k) * means[i][j]
                var = vars[i][j] * var / (vars[i][j] + var)
            
                for index in range(j):
                    weights[i][index] *= k
                weights[i][j] = 1-k

            outputs[i] = mean

        _, predicted = torch.max(outputs.data, 1)

        return predicted, weights

    def predicted_by_majority_voting(self, images, num_of_models):

        voting_pools = torch.zeros(num_of_models, images.shape[0], device=self.device, dtype=int)

        for i in range(num_of_models):
            self.nets[i].eval()
            outputs = self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))
            _, voting_pools[i] = torch.max(outputs.data, 1)     

        prediction, _ = torch.mode(voting_pools, 0)

        return prediction

    def obtained_stochastic_results_by_dropout(self, images, num_of_models, num_of_different_result):
        with torch.no_grad():
            
            predictions = torch.zeros(num_of_models, num_of_different_result, images.shape[0], len(self.classes), device=self.device)

            for i in range(num_of_models):
                self.nets[i].eval()

                predictions[i,0] = self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))
                
                self.nets[i].train()

                for j in range(1, num_of_different_result):
                    predictions[i,j] = self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))

            # swap dim from (num_of_model,num_of_different_try, num_of_image, num_of_class) to (num_of_images, num_of_model, num_of_different_try, num_of_class)
            predictions = torch.transpose(torch.transpose(predictions, 0, 2), 1, 2)

        return predictions

    def show_image(self, image):
        for i in range(self.num_of_models):
            plt.figure()
            plt.imshow(torch.moveaxis(self.transforms[i](deepcopy(image[None,:,:,:])).squeeze(), 0, -1 ).cpu())
            plt.title("model [{}/{}]".format(i+1, self.num_of_models))
        plt.show()
                
    def ransac(self):
        pass