# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ------------ system library ------------ #
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm

# ------------ custom library ------------ #
from conf import settings
from dataLoader import workerDataLoader, sourceDataLoader
from utils import get_network


class Worker:
    def __init__(self, _worker_id, _global_round, _args, _writer) -> None:
        self.worker_id = _worker_id
        self.args = _args
        self.model = get_network(_args)
        self.writer = _writer
        self.global_round = int(_global_round)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=_args.lr, momentum=0.9, weight_decay=5e-4)
        self.train_loader, self.test_loader = workerDataLoader(self.worker_id)
        # self.train_loader, self.test_loader = sourceDataLoader()

        if self.global_round != 0:
            print("{0} laod previous global round {1} model".format(self.worker_id, self.global_round-1))
            self.model.load_state_dict(torch.load("./"+settings.LOG_DIR+"/"+self.args.net+"/global_model/G"+str(_global_round-1)+"/aggregation.pt"))


    # Model training
    def train(self, save_model = True):
        print("Training Model ...")

        self.model.train()

        for epoch in range(1, settings.EPOCH+1):
            start = time.time()
            progress = tqdm(total=len(self.train_loader.dataset), ncols=100)

            for batch_index, (images, labels) in enumerate(self.train_loader):

                if self.args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                n_iter = (epoch - 1) * len(self.train_loader) + batch_index + 1

                progress.update(settings.BATCH_SIZE)
                self.writer.add_scalar('{0} Train/loss'.format(self.worker_id), loss.item(), n_iter)

            finish = time.time()

            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
            progress.close()
        
        if save_model:
            torch.save(self.model.state_dict(), settings.LOG_DIR+"/"+self.args.net+"/global_model/G"+str(self.global_round)+"/"+self.worker_id+".pt")


    # Evaluate model
    @torch.no_grad()
    def evaluate(self, tb=True):
        self.model.eval()

        test_loss = 0.0 # cost function error
        correct = 0.0

        for (images, labels) in self.test_loader:

            if self.args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

        print('Evaluating Model ...')
        print('Accuracy: {:.4f}, Average loss: {:.4f}'.format(correct.float() * 100 / len(self.test_loader.dataset), test_loss / len(self.test_loader.dataset)))

        # add informations to tensorboard
        if tb:
            self.writer.add_scalar('{0} Test/Average loss'.format(self.worker_id), test_loss / len(self.test_loader.dataset), self.global_round)
            self.writer.add_scalar('{0} Test/Accuracy'.format(self.worker_id), correct.float() * 100 / len(self.test_loader.dataset), self.global_round)

        return correct.float() * 100 / len(self.test_loader.dataset) 
    

    # Predcit model uncertainity
    def predictUncertainity(self, data, n_iter=100):
        output_list = []

        with torch.no_grad():
            for i in range(n_iter):
                output = self.model(data)
                output_list.append(output)
        output_mean = torch.mean(torch.stack(output_list), dim=0)
        output_std = torch.std(torch.stack(output_list), dim=0)

        return output_mean, output_std


    # Averaging uncertainity
    def avgUncertainty(self):
        images, labels = next(iter(self.test_loader))

        if self.args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        mean, std = self.predictUncertainity(images)
        mean_list, std_list = mean.tolist(), std.tolist()

        mean_avg = [round(sum([mean_list[column][row] for column in range(len(mean_list[0]))])/len(mean_list), 4) for row in range(len(mean_list[0]))]
        std_avg = [round(sum([std_list[column][row] for column in range(len(std_list[0]))])/len(std_list), 4) for row in range(len(std_list[0]))]
        
        print("Uncertainity \n   {0}".format(std_avg))

        return mean_avg, std_avg
        