import numpy as np
import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold


class snapsDataset(Dataset):
    """Dataset class for the onera and crm snapshot datasets"""

    def __init__(self, params, coeffs, device):
        self.coeffs = coeffs
        self.params = params
        self.device = device

    def __len__(self):
        return np.size(self.params, 0)

    def __getitem__(self, idx):
        inputs = tc.from_numpy(self.params[idx, :])
        targets = tc.from_numpy(self.coeffs[idx, :])
        return inputs.to(self.device), targets.to(self.device)


class podNetwork(nn.Module):
    def __init__(self, outSize):
        super(podNetwork, self).__init__()
        self.outSize = outSize
        self.arch0 = nn.Sequential(
            nn.Linear(2, 50),
            nn.Sigmoid(),
            nn.Linear(50, self.outSize)
        )
        self.arch1 = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, self.outSize)
        )

    def forward(self, x):
        x = self.arch1(x)
        return x


def main(inputs, targets, dimension, kfState, plots):
    """Function to train and return the neural network for predicting the POD coefficients. Inputs are the parameters
    used in the snapshots and the targets are the coefficients obtained from the snapshot matrix. dimension is the
    number of POD basis used. kfState is True/False to determine if the k-fold validation should be performed or not.
    Set kfState = True when tweaking the network and kfState = False when purely utilizing the network. Set plots
    parameter to false if training/testing error plots are not desired."""

    def train(net, trainLoader):
        net.train()
        currentLoss = 0
        for batchId, (inputs, targets) in enumerate(trainLoader):
            inputs, targets = inputs.to(device), targets.to(device)  # Send tensors to device
            optimizer.zero_grad()  # Ensures that the gradients are zeroed
            outputs = net(inputs.float())  # Calculates outputs of the network
            loss = lossFun(outputs, targets)  # Calculates the loss from the batch
            loss.backward()  # Back-propagates the loss through the network
            optimizer.step()  # Optimizes the parameters
            currentLoss += loss.item()  # Appends loss to keep track

        currentLoss = currentLoss / len(trainLoader.dataset)
        trainLoss.append(currentLoss)

    def test(net, testLoader):
        net.eval()
        currentLoss = 0
        with tc.no_grad():  # Do not compute gradients to save memory and time
            for batchId, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)  # Send tensors to device
                outputs = net(inputs.float())  # Calculate the outputs of the network
                loss = lossFun(outputs, targets)  # Calculate the losses
                currentLoss += loss.item()
        currentLoss = currentLoss / len(testLoader.dataset)
        testLoss.append(currentLoss)

    def resetWeights(net):
        # Used to reset model weights to avoid weight leakage in between folds
        for layer in net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    # misc setup and definitions
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')  # Uses cuda if available, otherwise uses cpu
    randomSeed = 0  # Sets random seed for repeatability
    tc.manual_seed(randomSeed)

    # Hyperparameters:
    hyperParamsList = [(1000, 1e-2, 2, 2), (1500, 5e-4, 15, 2)]  # Create a new tuple as for each new NN architecture
    hyperParams = 1
    epochs, lr, trainBatchSize, testBatchSize = hyperParamsList[hyperParams]
    k = 5  # Number of folds to use in the k-fold cross validation

    # Initializing the k-fold cross validation (to get more accurate performance/loss metrics)
    if kfState:
        training = []
        testing = []
        kf = KFold(n_splits=k, shuffle=True, random_state=randomSeed)
        for fold, (trainIds, testIds) in enumerate(kf.split(X=inputs, y=targets)):
            print('Fold ', fold)
            trainLoss = []
            testLoss = []

            # Separating training and testing samples by the Ids for the current fold and generating dataloaders
            trainData = snapsDataset(params=inputs[trainIds, :], coeffs=targets[trainIds, :], device=device)
            testData = snapsDataset(params=inputs[testIds, :], coeffs=targets[testIds, :], device=device)
            trainLoader = DataLoader(dataset=trainData, batch_size=trainBatchSize, shuffle=True)
            testLoader = DataLoader(dataset=testData, batch_size=testBatchSize, shuffle=True)

            network = podNetwork(outSize=dimension)
            network.apply(resetWeights)
            optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-3)
            lossFun = nn.L1Loss()

            # Training the network
            for epoch in range(1, epochs + 1):
                train(network, trainLoader)
                test(network, testLoader)

            training.append(trainLoss)
            testing.append(testLoss)

        training = np.array(training)
        testing = np.array(testing)
        avgTrainLoss = np.average(training, axis=0)
        avgTestLoss = np.average(testing, axis=0)

        if plots:
            # Plotting loss metrics
            fig1 = plt.figure()
            plt.plot(np.arange(0, epochs), avgTrainLoss,
                     label=f'Training, final iteration loss: {avgTrainLoss[-1]:.4f}')
            plt.plot(np.arange(0, epochs), avgTestLoss, label=f'Testing, final iteration loss: {avgTestLoss[-1]:.4f}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            fig1.show()

    # Performing final training on the network on the full dataset (no validation)
    network = podNetwork(outSize=dimension)
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-3)
    lossFun = nn.L1Loss()
    dataset = snapsDataset(params=inputs, coeffs=targets, device=device)
    datasetLoader = DataLoader(dataset, batch_size=trainBatchSize, shuffle=True)
    trainLoss = []
    for epoch in range(1, epochs + 1):
        train(network, datasetLoader)

    if plots:
        plt.plot(np.arange(0, epochs), trainLoss, label=f'Training, final iteration loss: {trainLoss[-1]:.4f}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Full dataset training losses')
        plt.legend(loc='upper right')
        plt.show()

    return network