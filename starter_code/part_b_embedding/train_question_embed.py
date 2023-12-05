import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt

from utils import load_train_question_correct_prop, load_subjects

from typing import Tuple, Dict

num_questions = 1774
out_shape = 1
batch_size = 1
embedding_size = 64
hidden_size = 32
epochs = 300
learning_rate = 0.01

question_correct_prop = load_train_question_correct_prop()

class QuestionEmbeddings(nn.Module):

    embedding: nn.Embedding
    lstm: nn.LSTM
    fc: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_questions, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_shape)

    def forward(self, input):
        embedded = self.embedding(input)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.mean(dim=1)
        fc_out = self.fc(lstm_out)
        output = F.tanh(fc_out)
        return output
    

class Data(Dataset):
    data: Dict[int, torch.Tensor]

    def __init__(self, data: Dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[int, torch.Tensor]:
        return index, self.data[index]


def _training(model: nn.Module, dataloader: DataLoader) -> None:
    
    model.train()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs), desc='epoch'):
        for index, data in dataloader:
            labels = torch.tensor(question_correct_prop[index]).to(torch.float32)
            preds = model(data).squeeze()

            loss = F.mse_loss(preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if epoch % 10 == 0:
            losses.append(eval(model, dataloader))

    plt.plot([i * 10 for i in range(epochs // 10)], losses)
    plt.show()


def eval(model: nn.Module, dataloader: DataLoader) -> float:

    loss = 0
    model.eval()
    with torch.no_grad():
        for index, data in dataloader:
            labels = torch.tensor(question_correct_prop[index]).to(torch.float32)
            preds = model(data).squeeze()

            loss += F.mse_loss(preds, labels).item()

    model.train()
    return loss


def load_data() -> DataLoader:
    subjects = load_subjects()
    data = Data(subjects)
    return DataLoader(data, batch_size=batch_size)


def main() -> None:

    torch.manual_seed(311)
    dataloader = load_data()

    model = QuestionEmbeddings()

    _training(model, dataloader)

    torch.save(model.state_dict(), 'embedding_param.pth')

        

if __name__ == '__main__':
    main()