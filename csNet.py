import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms.functional import InterpolationMode
from copy import deepcopy

# Defining the model

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class basic_cnn_t(nn.Module):
    def __init__(self, c1=96, c2= 192):
        super().__init__()
        d = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(d),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(d),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

    def forward(self, x):
        return self.m(x)

class basic_transform(nn.Module):

    def __init__(self, policy = None) -> None:
        super().__init__()

        if policy is None:
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
        
        self.policy = policy
    
    def forward(self, imgs):
        for i in range(imgs.shape[0]):
            for op_name, magnitude in self.policy:
                imgs[i] = _apply_op(imgs[i], op_name, magnitude, InterpolationMode.NEAREST, None)
    
        return imgs

class csNet(nn.Module):
    
    def __init__(self, num_of_models=16, device='cpu', transforms=None) -> None:
        super().__init__()

        self.device = device

        self.num_of_models = num_of_models

        if transforms is None:
            self.transforms = []
            self.transforms.append(basic_transform(policy=[("Identity", True)]))
            for _ in range(1, self.num_of_models):
                self.transforms.append(basic_transform())
        else:
            self.transforms = transforms

        self.nets = []
        self.lr_schedulers = []
        self.optimizers = []
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)    

        for i in range(self.num_of_models):
            self.nets.append(basic_cnn_t())
            self.optimizers.append(optim.SGD(self.nets[i].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5, nesterov=True))
            self.lr_schedulers.append(optim.lr_scheduler.ExponentialLR(optimizer=self.optimizers[i], gamma=0.95))
            self.nets[i].to(self.device)

    def train(self, train_loader, epochs):
        
        for net in self.nets:
            net.train()

        total_step = len(train_loader)

        for epoch in range(epochs):
            for j in range(self.num_of_models):
                total = 0
                correct = 0
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    total += labels.size(0)
                    outputs = self.nets[j](self.transforms[j](images))
                    loss = self.criterion(outputs, labels)
                    self.optimizers[j].zero_grad()
                    loss.backward()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    self.optimizers[j].step()
                    if (i+1) % 1000 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Model [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch+1, epochs, i+1, total_step, j+1, self.num_of_models, loss.item(), correct/total * 100))

            for j in range(self.num_of_models):
                self.lr_schedulers[j].step()

    def save_model(self, dir="pretrained_model", name="pre-trained_models.pth"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, name)
        
        nets = []
        transforms = []

        for net in self.nets:
            nets.append(net.state_dict())

        for transform in self.transforms:
            transforms.append(transform.policy)

        model = {"nets":nets,
                "transforms":transforms}

        torch.save(model, path)

        print("Saving models to:" + path)

    def save_optimizer(self, dir="pretrained_model", name="optimizer.pth"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, name)

        optimizers = []
        lr_schedulers = []

        for optimizer in self.optimizers:
            optimizers.append(optimizer.state_dict())

        for lr_scheduler in self.lr_schedulers:
            lr_schedulers.append(lr_scheduler.state_dict())

        optimizers_info = {"optimizers": optimizers, 
                            "lr_schedulers": lr_schedulers}
        
        torch.save(optimizers_info, path)

        print("Saving optimizers to:" + path)

    def load_models(self, path):
        
        print("Loading model from:" + path)

        models = torch.load(path)

        nets = models["nets"]

        transforms = models["transforms"]

        for i in range(self.num_of_models):
            self.nets[i].load_state_dict(nets[i])
            self.transforms[i].policy = transforms[i]

    def load_optimizers(self, path):

        print("Loading optimizer from:" + path)

        optimizers_info = torch.load(path)

        optimizers = optimizers_info["optimizers"]
        lr_schedulers = optimizers_info["optimizers"]

        for i in range(self.num_of_models):
            self.optimizers[i].load_state_dict(optimizers[i])
            self.lr_schedulers[i].load_state_dict(lr_schedulers[i])

    def test_each_model_accuracy_on_certain_dataset(self, test_loader, is_train_model=False):
        for i in range(self.num_of_models):
            if is_train_model:
                self.nets[i].train()
            else:
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
            print("Model:[{}/{}], Accuracy:{:.4f}%".format(i+1, self.num_of_models, correct/total * 100))
    
    def test_each_model_if_average_different_dropout_is_good(self, test_loader, num_of_test_for_each_model=16):
        for i in range(self.num_of_models):
            with torch.no_grad():
                correct = 0
                total = 0
                for j, (images, labels) in enumerate(test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.nets[i].eval()
                    outputs = self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))
                    
                    self.nets[i].train()
                    for _ in range(num_of_test_for_each_model):
                        outputs += self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))

                    outputs /= num_of_test_for_each_model

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Model:[{}/{}], Accuracy:{:.4f}%".format(i+1, self.num_of_models, correct/total * 100))

    def test_if_average_different_models_is_good(self, test_loader, is_train_model=False):
        for i in range(self.num_of_models):
            if is_train_model:
                self.nets[i].train()
            else:
                self.nets[i].eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            num_of_samples = len(test_loader)
            for j, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.softmax(self.nets[0](self.transforms[0](deepcopy(images))))
                for i in range(1,self.num_of_models):
                    outputs += self.softmax(self.nets[i](self.transforms[i](deepcopy(images))))

                outputs /= self.num_of_models
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print("Samples:[{}/{}], Accuracy:{:.4f}%".format(j+1, num_of_samples, correct/total * 100))

    def predict(self, images, num_of_test_for_each_model=16):
        
        images = images.to(self.device) 

        mini_batch_size = images.shape[0]

        num_of_class = 10

        with torch.no_grad():

            predictions_from_different_models = []   

            for i in range(self.num_of_models):
                predictions = []
                self.nets[i].eval()

                predictions.append(self.softmax(self.nets[i](self.transforms[i](deepcopy(images)))))
                
                self.nets[i].train()

                for _ in range(1, num_of_test_for_each_model):
                    predictions.append(self.softmax(self.nets[i](self.transforms[i](deepcopy(images)))))

                mini_batch_results = []

                for j in range(mini_batch_size):
                    predictions_for_one_image = torch.zeros((num_of_test_for_each_model, num_of_class), device=self.device)
                    
                    for k in range(num_of_test_for_each_model):
                        predictions_for_one_image[k] = predictions[k][j]

                    mean = torch.mean(predictions_for_one_image, dim=0)
                    cov = torch.cov(predictions_for_one_image.T)

                    mini_batch_results.append({"mean":mean, "cov":cov})

                predictions_from_different_models.append(mini_batch_results)

        fused_results = self.fuse_results_of_different_model(predictions_from_different_models)

        best_predictions = torch.zeros(mini_batch_size, device=self.device, dtype=int)

        for i in range(len(fused_results)):
            best_predictions[i] = torch.argmax(fused_results[i]["mean"])

        return best_predictions

    def fuse_results_of_different_model(self, predictions_from_different_models):
        """
        params:
            predictions_from_different_models: 
                List of predictions from different models.
                each elements in the list is also a list of mini_batch_results.
        """
        mini_batch_size = len(predictions_from_different_models[0])

        fused_results = []

        for j in range(mini_batch_size):
            mean = predictions_from_different_models[0][j]["mean"]
            cov = predictions_from_different_models[0][j]["cov"]
            for i in range(1, self.num_of_models):
                mean, cov = self.kalman_updates(mean, predictions_from_different_models[i][j]["mean"], cov, predictions_from_different_models[i][j]["cov"])
            fused_results.append({"mean":mean, "cov":cov})
        
        return fused_results

    def kalman_updates(self, mean_1, mean_2, cov_1, cov_2):
        K_1 = cov_2 @ torch.linalg.pinv(cov_1 + cov_2)
        K_2 = torch.eye(K_1.shape[0], device=self.device) - K_1
        mean = K_1 @ mean_1 +  K_2 @ mean_2
        cov = K_1 @ cov_1 @ K_1.T + K_2 @ cov_2 @ K_2.T
        return mean, cov  
                
    def ransac(self):
        pass