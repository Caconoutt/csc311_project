from matplotlib import pyplot as plt
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    
    zero_train_matrix[np.isnan(train_matrix)] = 0
   
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()
    #here is my update
        self.encoder = nn.Sequential(
            nn.Linear(num_question, 250),
            nn.ReLU(),
            nn.Linear(250, k),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 250),
            nn.ReLU(),
            nn.Linear(250, num_question),
        )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """

        g_w_norm = torch.norm(self.encoder.weight, 2)
        h_w_norm = torch.norm(self.decoder.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """

        encoded = self.encoder(inputs)
        out = self.decoder(encoded)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def plots(train_loss_list, valid_acc_list, k):
    """ save the figure training/validation accuracy/loss curves.
    :param train_acc_list: training accuracy data
    :param train_loss_list: training loss data
    :param valid_acc_list: validation accuracy data
    :param valid_loss_list: validation loss data
    :return: None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    data = np.array(valid_acc_list)
    ax1.plot(data[:, 0], data[:, 1], color='blue')
    ax1.set_title(f"Validation Accuracy of k = {k}")


    data = np.array(train_loss_list)
    ax2.plot(data[:, 0], data[:, 1], color="orange")
    ax2.set_title(f"Training Loss of k = {k}")
    plt.tight_layout()
    plt.show()


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,test_data,k):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    model.train()
    valid_acc_list = []
    train_loss_list = []
    max_ = 0

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    train_data_dict = load_train_csv('../data')

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        if epoch % 10 == 0:
            valid_accuracy, valid_loss = evaluate(model, zero_train_data, valid_data)
            train_acc, train_loss_test = evaluate(model, zero_train_data, train_data_dict)

            valid_acc_list.append((epoch, valid_accuracy))
            train_loss_list.append((epoch, train_loss_test))

            # test_acc, test_loss = evaluate(model, zero_train_data, test_data)
            # test_acc_list.append(test_acc)
            print("k = {},Epoch number: {} \tTraining Loss: {:.6f}\t "
                    "Valid Accuracy: {}".format(k, epoch, train_loss, valid_accuracy))

            if valid_accuracy > max_:
                max_ = valid_accuracy

    print(f'When k = {k},Max Valid Acc is {max_}')
   

    plots(train_loss_list, valid_acc_list, k)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    loss = 0.

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        target = valid_data["is_correct"][i]
        if guess == valid_data["is_correct"][i]:
            correct += 1
        loss_tensor = (output[0][valid_data["question_id"][i]] - target) ** 2.
        loss += loss_tensor.item()
        total += 1
    return correct / float(total), loss


def main():
    torch.set_printoptions(threshold=np.inf)
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Uncomment this to check out how we try out 5 different k and select the best k using the
    # validation set.

    # Set model hyperparameters.
    # for k in [10, 50, 100, 150, 200]:
    #     num_question = train_matrix.shape[1]
    #
    #     model = AutoEncoder(num_question=num_question, k=k)
    #
    #     # Set optimization hyperparameters.
    #     lr = 0.0002
    #     num_epoch = 100
    #     lamb = None
    #
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch,test_data,k)

    #####################################################################
    #                      Here is the output of optimal                    #
    #####################################################################
        # Here is the output for optimal.
    k = 100
    num_question = train_matrix.shape[1]
    model = AutoEncoder(num_question=num_question, k=k)
    lr = 0.0002
    num_epoch = 100
    lamb = None

    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch,test_data,k)

    # here is to report the test accuracy
    test = evaluate(model, zero_train_matrix, test_data)
    print(f"our test accuracy is {test}")

if __name__ == "__main__":
    main()