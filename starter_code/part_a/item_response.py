from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    for i, u_id in enumerate(data["user_id"]):
        #q_id = data["question_id"][i]
        log_lklihood += data["is_correct"][i]*np.log(sigmoid(theta[u_id]-beta[u_id])) + (1-data["is_correct"][i])*np.log(1-sigmoid(theta[u_id]-beta[u_id]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    u_id_arr = np.array(data["user_id"])
    q_id_arr = np.array(data["question_id"])
    c_id_arr = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    for i in range(len(theta)):
        theta[i] -= lr * (np.sum(sigmoid(theta_copy[i] - beta_copy)[q_id_arr[u_id_arr == i]]) - np.sum(c_id_arr[u_id_arr == i]))

    for j in range(len(beta)):
        beta[j] -= lr * (np.sum(c_id_arr[q_id_arr == j]) - np.sum(sigmoid(theta_copy - beta_copy[j])[u_id_arr[q_id_arr == j]]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst, train_acc_lst = [], []
    val_log_likelihood, train_log_likelihood = [], []

    for i in range(iterations):
        # Log likelihood
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_log_likelihood.append(train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("Negative Loglikelihood: {} \t Train Score: {} \t Validation Score: {}".format(train_neg_lld, train_score, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # (b) tune hyperparameters and plot neg loglike against iteration
    lr = 0.01
    iterations = 10
    theta, beta, val_loglike, train_loglike = irt(train_data, val_data,lr, iterations)
    plt.plot(val_loglike, marker = 'o', label="validation")
    plt.plot(train_loglike, marker = 'x',label="train")
    plt.xlabel('Number of iterations')
    plt.ylabel('Negative LogLikelihood')
    plt.legend()

    # (c) report validation and test accuracy
    print("The final validation accuracy is {}\nThe final test accuracy is {}".format(evaluate(val_data, theta, beta),evaluate(test_data, theta, beta)))

    # (d) plot the probability of correctness against theta for three selected questions
    selected_q = [111,222,999]
    theta.sort()
    for q in selected_q:
        p = sigmoid(theta - beta[q])
        plt.plot(theta.T,p, label=f"Selected question {q}")
    plt.xlabel('Theta')
    plt.ylabel('Probability of correct answer on selected question')
    plt.legend()


if __name__ == "__main__":
    main()


