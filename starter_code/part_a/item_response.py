from utils import *

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
    log_lklihood = 0.
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
    pass
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
    theta = None
    beta = None

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()




######Archive
def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: matrix 542 x 1774
    :param theta: Vector 1 x 542
    :param beta: Vector  1 x 1774
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    #sparse_matrix = load_train_sparse("data")

    # clean data matrix
    c = data.astype(int)
    #value of nan after converted to float
    c[c==-9223372036854775808] = 0

    theta = theta.reshape((542,1))
    a = np.tile(theta, (1,1774)) # 542 x 1774

    b = np.tile(beta, (542,1)) # 542 x 1774
    sig = sigmoid(a - b)

    log_sig = np.log(sig, out=np.zeros_like(sig), where=(sig!=0))
    log_1_minus_sig = np.log((1-sig), out=np.zeros_like(sig), where=(sig!=0))

    log_lklihood = np.trace(np.dot(data, log_sig.T)) + np.trace(np.dot((1-data), log_1_minus_sig.T))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood



#newest nov.26 10pm
data = load_train_sparse("data")

N,M = data.shape
theta = np.zeros((1,N))
beta = np.zeros((1,M))

# clean data matrix
denseData = data.todense()

theta = theta.reshape((542,1))
a = np.tile(theta, (1,1774)) # 542 x 1774

b = np.tile(beta, (542,1)) # 542 x 1774
sig = sigmoid(a - b)
log_sig = np.log(sig, out=np.zeros_like(sig), where=(sig!=0))
log_1_minus_sig = np.log((1-sig), out=np.zeros_like(sig), where=(sig!=0))

t1 = np.multiply(denseData, log_sig)
t2 = np.multiply(1-denseData, log_1_minus_sig)
loglikelihood = np.sum(t1+t2)
print(loglikelihood)
#print(c_times_sigma.shape)