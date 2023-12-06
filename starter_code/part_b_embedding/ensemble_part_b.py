import numpy as np
from tqdm import tqdm

from train_question_embed import QuestionEmbeddings

import time
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from utils import load_train_csv, load_valid_csv, load_public_test_csv
from typing import Tuple, List

in_shape = 2
out_shape = 1
learning_rate = 0.01
batch_size = 128

class LogisticRegression(nn.Module):

    linear: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_shape, out_shape)

        init.zeros_(self.linear.weight)

    def forward(self, input) -> torch.Tensor:
        logits = self.linear(input)
        output = F.sigmoid(logits)
        return output
    

class NeuralNetwork(nn.Module):
    
    model: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_shape)
        )

    def forward(self, input) -> torch.Tensor:
        logits = self.model(input)
        output = F.sigmoid(logits)
        return output


class Data(Dataset):
    data: torch.Tensor
    labels: torch.Tensor

    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        return self.data[index], self.labels[index]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = load_train_csv()
    val = load_valid_csv()
    test = load_public_test_csv()

    embedding_nn = QuestionEmbeddings()
    embedding_nn.load_state_dict(torch.load('embedding_param.pth'))

    train_question_id = torch.tensor(train['question_id']).reshape(-1, 1)
    train_question_embedded = embedding_nn(train_question_id).flatten().detach().numpy()

    train = np.array([train_question_embedded, train['user_id'], train['is_correct']]).T

    val_question_id = torch.tensor(val['question_id']).reshape(-1, 1)
    val_question_embedded = embedding_nn(val_question_id).flatten().detach().numpy()
    val_data = np.array([val_question_embedded, val['user_id']]).T
    val_labels = np.array(val['is_correct'])

    test_question_id = torch.tensor(test['question_id']).reshape(-1, 1)
    test_question_embedded = embedding_nn(test_question_id).flatten().detach().numpy()
    test_data = np.array([test_question_embedded, test['user_id']]).T
    test_labels = np.array(test['is_correct'])

    return train, val_data, val_labels, test_data, test_labels


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


def _training(model: nn.Module, dataloader: DataLoader, epochs: int) -> None:

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    accuracies = []

    for epoch in tqdm(range(epochs), desc="epoch"):
        model.train()
        for data, labels in dataloader:
            probs = model(data).squeeze()
            loss = F.binary_cross_entropy(probs, labels, reduction='sum')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if epoch % 10 == 0:
            loss, accuracy = evaluate(model, dataloader)
            losses.append(loss)
            accuracies.append(accuracy)
    
    epoch_indices = [i * 10 for i in range(epochs // 10)]
    plt.clf()
    plt.plot(epoch_indices, losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{model._get_name()}_loss.png', dpi=300)

    plt.clf()
    plt.plot(epoch_indices, accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'{model._get_name()}_accuracy.png', dpi=300)

    
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:

    model.eval()
    count = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for data, labels in dataloader:
            probs = model(data).squeeze()
            loss += F.binary_cross_entropy(probs, labels, reduction='sum').item()
            preds = (probs >= 0.5).to(torch.float32)
            
            count += torch.sum(preds == labels).item()
            total += len(labels)

    accuracy = count / total
    return loss, accuracy


def train_eval(model: nn.Module, train_data: np.ndarray,
               train_labels: np.ndarray, val_data: np.ndarray,
               val_labels: np.ndarray, epochs: int):
    training_dataset = Data(train_data, train_labels)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    
    _training(model, training_dataloader, epochs)

    val_dataset = Data(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    training_loss, training_accuracy = evaluate(model, training_dataloader)
    print(f'Training loss and accuracy for {model._get_name()}: {training_loss}, {training_accuracy}')

    val_loss, val_accuracy = evaluate(model, val_dataloader)
    print(f'Validation loss and accuracy for {model._get_name()}: {val_loss}, {val_accuracy}')



def ensemble(models: List[nn.Module], dataloader: DataLoader) -> torch.Tensor:
    all_preds = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            model_preds = torch.tensor([])
            for data, _ in dataloader:
                probs = model(data).squeeze()
                preds = (probs > 0.5).to(torch.float32)
                
                model_preds = torch.cat((model_preds, preds))
            all_preds.append(list(model_preds))

    sum_preds = torch.sum(torch.tensor(all_preds), dim=0)
    ensemble_preds = (sum_preds >= 2).to(torch.float32)

    return ensemble_preds
        

def main() -> None:
    train, val_data, val_labels, test_data, test_labels = load_data()
    bootstrap_samples = bootstrap(train)

    torch.manual_seed(311)



    # logistic_model = LogisticRegression()
    # logistic_data, logistic_labels = bootstrap_samples[0]
    # train_eval(logistic_model, logistic_data, logistic_labels, val_data,val_labels, 50)

    # torch.save(logistic_model.state_dict(), 'logistic_model.pth')

    # neural_network1 = NeuralNetwork()
    # neural_network_data1, neural_network_labels1 = bootstrap_samples[1]
    # train_eval(neural_network1, neural_network_data1, neural_network_labels1, val_data, val_labels, 200)

    # torch.save(neural_network1.state_dict(), 'neural_network1.pth')

    # neural_network2 = NeuralNetwork()
    # neural_network_data2, neural_network_labels2 = bootstrap_samples[2]
    # train_eval(neural_network2, neural_network_data2, neural_network_labels2, val_data, val_labels, 200)

    # torch.save(neural_network2.state_dict(), 'neural_network2.pth')

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


