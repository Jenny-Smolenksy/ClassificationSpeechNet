from __future__ import print_function
import argparse
import torch.optim as optim
import Datasets
from Datasets import ClassificationLoader
import numpy as np
import torch.nn.functional as F
from model import VGG
import os
import torch


def build_model_name(args):
    args_dict = ["optimizer", "lr", "batch_size", "arc", "class_num"]
    full_name = ""
    for arg in args_dict:
        full_name += str(arg) + "_" + str(getattr(args, arg)) + "_"

    return full_name[:-1] + ".pth"


class ClassificationNet:

    def __init__(self, data_folder, num_classes, num_of_workers):
        """
        Construction to initialize net
        Args:
            data_folder (): folder path, contains folder train and valid
            num_classes (): num of classes for multiclass classification
            num_of_workers (): for pytorch
        """
        print('initializing classification net')
        # set args
        self.args = self.init_args(data_folder, num_classes)
        # set data
        self.train_loader, self.valid_loader = self.init_data_loaders(num_of_workers)
        # create model
        self.model = self.create_model()
        # create optimizer
        self.optimizer = self.create_optimizer(self.model)
        # load previous model, if exists
        prev_model = self.load_previous_model()
        if prev_model:
            self.model = prev_model

    def init_args(self, data_path, num_classes):
        """
        create arguments for net
        Args:
            data_path (): folder path for data
            num_classes (): multiclass classification

        Returns:
            args
        """

        parser = argparse.ArgumentParser(
            description='ConvNets for Speech Commands Recognition')
        parser.add_argument('--train_path', default=data_path + "\\train",
                            help='path to the train data folder')
        parser.add_argument('--valid_path', default=data_path + "\\valid",
                            help='path to the valid data folder')
        parser.add_argument('--batch_size', type=int, default=32,
                            metavar='N', help='training and valid batch size')
        parser.add_argument('--test_batch_size', type=int, default=32,
                            metavar='N', help='batch size for testing')
        parser.add_argument('--arc', default='VGG11',
                            help='network architecture: VGG11, VGG13, VGG16, VGG19')
        parser.add_argument('--epochs', type=int, default=30,
                            metavar='N', help='number of epochs to train')
        parser.add_argument('--lr', type=float, default=0.001,
                            metavar='LR', help='learning rate')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum, for SGD only')
        parser.add_argument('--optimizer', default='adam',
                            help='optimization method: sgd | adam')
        parser.add_argument('--cuda', default=True, help='enable CUDA')
        parser.add_argument('--seed', type=int, default=1234,
                            metavar='S', help='random seed')
        parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                            help='num of batches to wait until logging train status')
        parser.add_argument('--patience', type=int, default=3, metavar='N',
                            help='how many epochs of no loss improvement should we wait before stop training')
        parser.add_argument('--max_len', type=int, default=101,
                            help='window size for the stft')
        parser.add_argument('--window_size', default=.02,
                            help='window size for the stft')
        parser.add_argument('--window_stride', default=.01,
                            help='window stride for the stft')
        parser.add_argument('--window_type', default='hamming',
                            help='window type for the stft')
        parser.add_argument('--normalize', default=True,
                            help='boolean, wheather or not to normalize the spect')
        parser.add_argument('--save_folder', type=str,
                            default='trained_model/',
                            help='path to save the final model')
        # change here number of classes
        parser.add_argument('--class_num', type=int, default=num_classes,
                            help='number of classes to classify')
        parser.add_argument('--prev_classification_model', type=str,
                            default='trained_model/optimizer_adam_lr_0.001_batch_size_32_arc_VGG11_class_num_6.pth',
                            help='the location of the prev classification model')
        args = parser.parse_args()
        args.cuda = args.cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.manual_seed(args.seed)
        return args

    def init_data_loaders(self, num_of_workers):
        """
        create torch data loaders for train and validation data
        Args:
            num_of_workers ():

        Returns:
            train , validation data loaders
        """

        train_dataset = ClassificationLoader(self.args.train_path,
                                             window_size=self.args.window_size,
                                             window_stride=self.args.window_stride,
                                             window_type=self.args.window_type,
                                             normalize=self.args.normalize,
                                             max_len=self.args.max_len)
        sampler_train = Datasets.ImbalancedDatasetSampler(train_dataset)

        train_loader = \
            torch.utils.data.DataLoader(train_dataset,
                                        batch_size=self.args.batch_size, shuffle=None,
                                        num_workers=num_of_workers, pin_memory=self.args.cuda,
                                        sampler=sampler_train)

        valid_dataset = ClassificationLoader(self.args.valid_path,
                                             window_size=self.args.window_size,
                                             window_stride=self.args.window_stride,
                                             window_type=self.args.window_type,
                                             normalize=self.args.normalize,
                                             max_len=self.args.max_len)

        valid_loader = \
            torch.utils.data.DataLoader(valid_dataset, batch_size=self.args.batch_size,
                                        shuffle=None, num_workers=num_of_workers,
                                        pin_memory=self.args.cuda, sampler=None)

        return train_loader, valid_loader,

    def create_model(self):
        """
        create the model
        Returns:
            vgg model
        """
        # build model
        if self.args.arc.startswith("VGG"):
            model = VGG(self.args.arc, self.args.class_num)
        else:
            model = VGG("VGG11", self.args.class_num)

        if self.args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        return model

    def create_optimizer(self, model):
        """
        create optimizer for the model
        Args:
            model ():

        Returns:
            torch optimizer
        """
        # define optimizer
        if self.args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum)
        return optimizer

    def load_previous_model(self):
        """
        check for model
        Returns:
            VGG previoud model if exists, none if no such model.
        """
        model = None
        # build model
        if os.path.isfile(self.args.prev_classification_model):  # model exists
            model, check_acc, check_epoch, class_num = \
                ClassificationNet.load_model_params(self.args.prev_classification_model, self.model)
            print(f"found trained model, prev valid loss: {check_acc}, after {check_epoch} epochs")
            if class_num != self.args.class_num:
                raise Exception("saved model num of classes in target different from current")

        return model

    @staticmethod
    def load_model_params(save_dir, model):
        """
        load existing model parameters
        Args:
            save_dir (): path of the existing model
            model (): type to search

        Returns:
            trained model, previous: validation loss, epoch number, classification count
        """
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)

        current_dict = model.state_dict()
        saved_values = list(checkpoint['net'].values())
        index = 0
        for key, val in current_dict.items():
            current_dict[key] = saved_values[index]
            index += 1

        model.load_state_dict(current_dict)

        return model, checkpoint['acc'], checkpoint['epoch'], checkpoint['class_num']

    def train(self):
        """
        train the model
        Returns:

        """
        best_valid_loss = np.inf
        iteration = 0
        epoch = 1
        print('training model..')
        # training with early stopping
        while (epoch < self.args.epochs + 1) and (iteration < self.args.patience):

            ClassificationNet.train_classification(self.train_loader, self.model, self.optimizer,
                                                   epoch, self.args.cuda, self.args.log_interval)

            valid_loss, acc = \
                ClassificationNet.test_classification(self.valid_loader, self.model, self.args.cuda)

            if valid_loss > best_valid_loss:
                iteration += 1
                print('Loss was not improved, iteration {0}'.format(str(iteration)))
            else:
                print('Saving model...')
                iteration = 0
                best_valid_loss = valid_loss

                # model = nn.DataParallel(model)  # jenny added this
                state = {
                    'net': self.model.state_dict(),
                    'acc': valid_loss,
                    'epoch': epoch,
                    'class_num': self.args.class_num
                }

                if not os.path.isdir(self.args.save_folder):
                    os.mkdir(self.args.save_folder)

                torch.save(state, self.args.save_folder + '/' + build_model_name(self.args))
            epoch += 1

    def test(self, folder_path, num_of_workers=0):
        """
        test the model on given test set
        Args:
            folder_path (): for test set
            num_of_workers (): for torch

        Returns:

        """

        test_dataset = ClassificationLoader(folder_path,
                                            window_size=self.args.window_size,
                                            window_stride=self.args.window_stride,
                                            window_type=self.args.window_type,
                                            normalize=self.args.normalize,
                                            max_len=self.args.max_len)
        test_loader = \
            torch.utils.data.DataLoader(test_dataset, batch_size=self.args.test_batch_size,
                                        shuffle=None, num_workers=num_of_workers,
                                        pin_memory=self.args.cuda)

        ClassificationNet.test_classification(test_loader, self.model, self.args.cuda)

    @staticmethod
    def test_classification(loader, model, cuda, verbose=True):
        """
        classification model tester
        Args:
            loader (): torch data loader
            model (): trained model
            cuda ():
            verbose (): to print results

        Returns:

        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                if cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(loader.dataset)
            if verbose:
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        return test_loss, 100. * correct / len(loader.dataset)

    @staticmethod
    def train_classification(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
        """
        classification model trainer
        Args:
            loader (): torch data loader
            model (): to train
            optimizer (): for training process
            epoch (): number of epochs to run
            cuda (): True/False
            log_interval (): time interval to log results
            verbose (): to print progress

        Returns:
            average loss
        """
        model.train()
        global_epoch_loss = 0
        for batch_idx, (data, target) in enumerate(loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print("pred ", pred)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            global_epoch_loss += loss.item()
            if verbose:
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(loader.dataset), 100.
                               * batch_idx / len(loader), loss.item()))
        return global_epoch_loss / len(loader.dataset)


def main():
    # main folder, in it: train folder, valid folder
    data_path = "data_folder_path"

    net = ClassificationNet(data_path, num_classes=6, num_of_workers=0)
    net.train()

    test_folder = "test_path"
    net.test(test_folder)


if __name__ == '__main__':
    main()
