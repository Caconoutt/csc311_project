import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

from train_question_embed import QuestionEmbeddings

import time
import torch
import matplotlib.pyplot as plt

from utils import load_train_csv, load_valid_csv, load_public_test_csv
from typing import Tuple, List

def load_data() -> Tuple[np.ndarray]:
    train = load_train_csv()
    val = load_valid_csv()
    test = load_public_test_csv()

    embedding_nn = QuestionEmbeddings()
    embedding_nn.load_state_dict(torch.load('embedding_param.pth'))

    train_question_id = torch.tensor(train['question_id']).reshape(-1, 1)
    train_question_embedded = embedding_nn(train_question_id).flatten().detach().numpy()
    train_data = np.array([train_question_embedded, train['user_id']]).T
    train_labels = np.array(train['is_correct'])

    val_question_id = torch.tensor(val['question_id']).reshape(-1, 1)
    val_question_embedded = embedding_nn(val_question_id).flatten().detach().numpy()
    val_data = np.array([val_question_embedded, val['user_id']]).T
    val_labels = np.array(val['is_correct'])

    test_question_id = torch.tensor(test['question_id']).reshape(-1, 1)
    test_question_embedded = embedding_nn(test_question_id).flatten().detach().numpy()
    test_data = np.array([test_question_embedded, test['user_id']]).T
    test_labels = np.array(test['is_correct'])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def train(train_data: np.ndarray, train_labels: np.ndarray, k: int) -> KNeighborsClassifier:

    knn_model = KNeighborsClassifier(k, weights='uniform')

    knn_model.fit(train_data, train_labels)

    return knn_model

def knn(train_data: np.ndarray, train_labels: np.ndarray,
        val_data: np.ndarray, val_labels: np.ndarray,
        k: int, transformer: FunctionTransformer) -> float:
    
    model = train(train_data, train_labels, k)
    accuracy = evaluate(model, transformer, val_data, val_labels)

    return accuracy

def evaluate(model: KNeighborsClassifier, transformer: FunctionTransformer,
              data: np.ndarray, labels: np.ndarray) -> float:

    data = transformer.transform(data)
    preds = model.predict(data)
    accuracy = accuracy_score(labels, preds)
    return accuracy

def main() -> None:
    transformed_train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()
    accuracies = []
    k = [5 * i + 1 for i in range(11)]

    transformer = FunctionTransformer()
    transformed_train_data = transformer.fit_transform(transformed_train_data)
    for i in k:
        accuracy = knn(transformed_train_data, train_labels, val_data, val_labels, i, transformer)
        accuracies.append(accuracy)

    plt.plot(k, accuracies)
    plt.show()

    

    best_k = 51
    best_model = train(transformed_train_data, train_labels, best_k)
    val_accuracy = evaluate(best_model, transformer, val_data, val_labels)
    test_accuracy = evaluate(best_model, transformer, test_data, test_labels)

    print(f'Validation accuracy: {val_accuracy}')
    print(f'Test accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()