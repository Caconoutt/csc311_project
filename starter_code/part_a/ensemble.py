import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from utils import load_train_csv, load_valid_csv, load_public_test_csv
from typing import Tuple, List


in_shape = 2
out_shape = 2
batch_size = 128
learning_rate = 0.01


class LogisticRegression(nn.Module):

    model: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_shape, out_shape),
            nn.Sigmoid()
        )

    def forward(self, input) -> torch.Tensor:
        output = self.model(input)
        return output
    

class NeuralNetwork(nn.Module):
    model: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_shape),
            nn.Sigmoid()
        )

    def forward(self, input) -> torch.Tensor:
        output = self.model(input)
        return output


class Data(Dataset):
    data: torch.Tensor
    labels: torch.Tensor

    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        return self.data[index], self.labels[index]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = load_train_csv()
    val = load_valid_csv()
    test = load_public_test_csv()

    train = np.array([train['user_id'], train['question_id'], train['is_correct']]).T
    val_data = np.array([val['user_id'], val['question_id']]).T
    val_labels = np.array(val['is_correct'])
    test_data = np.array([test['user_id'], test['question_id']]).T
    test_labels = np.array(test['is_correct'])

    return train, val_data, val_labels, test_data, test_labels


def training(model: nn.Module, dataloader: DataLoader, epochs: int) -> None:

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        count = 0
        total = 0
        for data, labels in dataloader:
            data, labels = data, labels
            logits = model(data)
            loss = F.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            count += torch.sum(preds == labels)
            total += len(labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        accuracy = count / total
        print(f'Loss after epoch {epoch + 1} for {model._get_name()}: {loss.item()}')
        print(f'Training accuracy after epoch {epoch + 1} for {model._get_name()}: {accuracy}')

    
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:

    model.eval()
    count = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataloader:
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            count += torch.sum(preds == labels)
            total += len(labels)

    accuracy = count / total
    return accuracy


def bootstrap(train: np.ndarray, num_bootstraps=3) -> List[Tuple[np.ndarray, np.ndarray]]:
    bootstrap_samples = []
    n = train.shape[0]
    rng = np.random.default_rng(311)
    for _ in range(num_bootstraps):
        bootstrap_sample = rng.choice(train, size=n, replace=True)
        train_data = bootstrap_sample[:, :2]
        train_labels = bootstrap_sample[:, 2]

        bootstrap_samples.append((train_data, train_labels))
    
    return bootstrap_samples


def train_eval(model: nn.Module, train_data: np.ndarray,
               train_labels: np.ndarray, val_data: np.ndarray,
               val_labels: np.ndarray, epochs: int):
    training_dataset = Data(train_data, train_labels)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    
    training(model, training_dataloader, epochs)

    val_dataset = Data(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print(f'Training accuracy for {model._get_name()}: {evaluate_accuracy(model, training_dataloader)}')
    print(f'Validation accuracy for {model._get_name()}: {evaluate_accuracy(model, val_dataloader)}')


def ensemble(models: List[nn.Module], dataloader: DataLoader) -> torch.Tensor:
    all_preds = []
    
    for model in models:
        model_preds = torch.tensor([])
        model.eval()
        for data, _ in dataloader:
            with torch.no_grad():
                logits = model(data)
                preds = torch.argmax(logits, dim=1)
                model_preds = torch.cat((model_preds, preds), dim=0)

        all_preds.append(list(model_preds))

    sum_preds = torch.sum(torch.tensor(all_preds), dim=0)
    ensemble_preds = sum_preds > 1.5

    return ensemble_preds
        

def main() -> None:
    train, val_data, val_labels, test_data, test_labels = load_data()
    bootstrap_samples = bootstrap(train)

    torch.manual_seed(311)

    logistic_model = LogisticRegression()
    logistic_data, logistic_labels = bootstrap_samples[0]
    train_eval(logistic_model, logistic_data, logistic_labels, val_data, val_labels, 10)

    torch.save(logistic_model.state_dict(), 'logistic_model.pth')

    neural_network1 = NeuralNetwork()
    neural_network_data1, neural_network_labels1 = bootstrap_samples[1]
    train_eval(neural_network1, neural_network_data1, neural_network_labels1, val_data, val_labels, 50)

    torch.save(neural_network1.state_dict(), 'neural_network1.pth')

    neural_network2 = NeuralNetwork()
    neural_network_data2, neural_network_labels2 = bootstrap_samples[2]
    train_eval(neural_network2, neural_network_data2, neural_network_labels2, val_data, val_labels, 50)

    torch.save(neural_network2.state_dict(), 'neural_network2.pth')

    logistic_model = LogisticRegression()
    logistic_model.load_state_dict(torch.load('logistic_model.pth'))

    neural_network1 = NeuralNetwork()
    neural_network1.load_state_dict(torch.load('neural_network1.pth'))

    neural_network2 = NeuralNetwork()
    neural_network2.load_state_dict(torch.load('neural_network2.pth'))

    models = [logistic_model, neural_network1, neural_network2]

    val_dataset = Data(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = Data(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    val_preds = ensemble(models, val_dataloader)
    test_preds = ensemble(models, test_dataloader)

    val_accuracy = torch.mean((val_preds == torch.tensor(val_labels)).to(torch.float32))
    test_accuracy = torch.mean((test_preds == torch.tensor(test_labels)).to(torch.float32))

    print(f'Ensemble validation accuracy: {val_accuracy}')
    print(f'Ensemble test accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()


