from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
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
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
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

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # DONE:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        encoded = F.sigmoid(self.g(inputs))
        out = F.sigmoid(self.h(encoded))
        # out = inputs
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
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
    # DONE: Add a regularizer to the cost function.
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    training_loss = []
    validation_accuracy = []
    x = []
    for epoch in range(0, num_epoch+1):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # Here is what I put for regularizer!
            l2_reg = model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.) + (lamb/2.) * l2_reg
            loss.backward()

            train_loss += loss.item()
            optimizer.step()
        if epoch % 50 == 0:
            x.append(epoch)
            training_loss.append(train_loss)
            valid_accuracy = evaluate(model, zero_train_data, valid_data)

            validation_accuracy.append(valid_accuracy)
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Accuracy: {}".format(epoch, train_loss, valid_accuracy))

    return validation_accuracy, training_loss, x


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

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # DONE:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_options = [10, 50, 100, 200, 500]
    # k_options = 50
    # Set optimization hyperparameters.
    lr = 0.005
    # lr_options = [0.001, 0.005, 0.01, 0.1]
    # lr_options = [0.005, 0.01, 0.02, 0.05]
    # num_epoch = [100, 250, 500]

    lamb = 0.0

    dimension = train_matrix.shape[1]

    # this is for Q3(c), picking a k with highest validation accuracy, with same epoch number 200.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    colors = ['crimson', 'darkorange', 'green', 'purple', 'blue', 'tomato', 'gold',
          ]
    i = 0
    for k in k_options:

        epoch_for_k = 500
        model = AutoEncoder(dimension, k)
        val_acc, tr_loss,x = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epoch_for_k)

        #x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        #x = [0, 25, 50]
        color = colors[i]
        ax1.plot(x, val_acc, label=f'k={k}', color=color)
        ax2.plot(x, tr_loss, label=f'k={k}', color=color)
        i += 1
    #this plot tutorial, I found on: https://www.geeksforgeeks.org/matplotlib-axes-axes-set_xlabel-in-python/ and combined my personal experience
    ax1.set_xlabel('Epoch Value')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Validation Accuracy for various k values, with lr = 0.005', fontweight ='bold')
    ax1.grid(True)
    ax1.legend()
    #
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.grid(True)
    ax2.set_title('Training Loss for various k values, with lr = 0.005',fontweight ='bold')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # this is for Q3(c), with k*, displaying learning rate influcence on validatino accuracy.
    # k_best = 50
    # fig, ax1 = plt.subplots(3, 2, figsize=(12, 7))
    # colors = ['crimson', 'green', 'darkorange', 'purple', 'blue', 'tomato', 'gold',
    #        ]
    #
    # row = 0
    #
    # for epoch in num_epoch:
    #     #x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    #     i = 0
    #     for lr in lr_options:
    #         model = AutoEncoder(dimension, k_best)
    #         val_acc, tr_loss, x = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epoch)
    #
    #         color = colors[i]
    #         ax1[row][0].plot(x, val_acc, label=f'lr={lr}', color=color)
    #         ax1[row][1].plot(x, tr_loss, label=f'lr={lr}', color=color)
    #         i += 1
    # # this plot tutorial, I found on: https://www.geeksforgeeks.org/matplotlib-axes-axes-set_xlabel-in-python/ and combined my personal experience
    #     ax1[row][0].set_ylabel('Validation Accuracy')
    #     ax1[row][0].set_xlabel('Epoch Value')
    #     ax1[row][0].set_title(f'Validation Accuracy for various learning rate, with epoch number = {epoch}')
    #     # ax1[row][0].grid(True)
    #     ax1[row][0].legend()
    #
    #     ax1[row][1].set_ylabel('Training Loss')
    #     ax1[row][1].set_xlabel('Epoch')
    #     ax1[row][1].set_title(f'Training Loss for various learning rate, with epoch number ={epoch}')
    #     # ax1[row][1].grid(True)
    #     ax1[row][1].legend()
    #     row += 1
    #
    # plt.tight_layout()
    # plt.show()


    # this is for Q3(d), find test accuracy:
    # lamb = 0.0
    # epoch_for_k = 250
    # lr = 0.005
    # k = 10
    # model = AutoEncoder(dimension, k)
    # val_acc, tr_loss, x = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epoch_for_k)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    # # x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ax1.plot(x, val_acc, color='blue')
    # ax1.set_title("Validation Accuracy under optimal")
    # ax2.plot(x, tr_loss, color='green')
    # ax2.set_title("Training loss under optimal")
    # plt.tight_layout()
    # plt.show()

    # here is to report the test accuracy
    test = evaluate(model, zero_train_matrix, test_data)
    print(f"our test accuracy is {test}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
