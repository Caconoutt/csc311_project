from scipy.sparse import load_npz
import pandas as pd
import numpy as np
import csv
import os


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_train_sparse(root_dir="/data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_train_csv(root_dir="/data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_valid_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="/data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="/data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions


def classify_subject():
    pass


def question_mapping(question_id, question_subject: pd.DataFrame, if_max=False):
    subject_id = question_subject[question_subject['question_id'] == question_id]['subject_id'].item()[1:-1].split(', ')
    subject_id = list(map(int, subject_id))[-2:]
    return subject_id


def combine_data(origin_train_data_position: str, question_to_subject_data_position: str,
                 origin_valid_data_position: str, origin_test_data_position: str,if_max=False):

    stu_question_accuracy_train_data = pd.read_csv(origin_train_data_position)
    stu_question_accuracy_train_data = stu_question_accuracy_train_data.sort_values(by='user_id')

    stu_question_accuracy_valid_data = pd.read_csv(origin_valid_data_position)
    stu_question_accuracy_valid_data = stu_question_accuracy_valid_data.sort_values(by='user_id')

    stu_question_accuracy_test_data = pd.read_csv(origin_test_data_position)
    stu_question_accuracy_test_data = stu_question_accuracy_test_data.sort_values(by='user_id')

    question_subject = pd.read_csv(question_to_subject_data_position)
    train_data_list = list(map(lambda x: question_mapping(x, question_subject,if_max=if_max),
                                                              stu_question_accuracy_train_data['question_id']))


    valid_data_list = list(map(lambda x: question_mapping(x, question_subject,if_max=if_max),
                                                              stu_question_accuracy_valid_data['question_id']))

    test_data_list = list(map(lambda x: question_mapping(x, question_subject, if_max=if_max),
                               stu_question_accuracy_test_data['question_id']))

    for numb in range(2):
        stu_question_accuracy_train_data[f'subject_id{numb + 1}'] = list(map(lambda x: x[numb], train_data_list))
        stu_question_accuracy_valid_data[f'subject_id{numb + 1}'] = list(map(lambda x: x[numb], valid_data_list))
        stu_question_accuracy_test_data[f'subject_id{numb + 1}'] = list(map(lambda x: x[numb], test_data_list))

    stu_question_accuracy_train_data = stu_question_accuracy_train_data[['question_id', 'user_id', 'subject_id1',
                                                                         'subject_id2', 'is_correct']]

    stu_question_accuracy_valid_data = stu_question_accuracy_valid_data[['question_id', 'user_id', 'subject_id1',
                                                                         'subject_id2', 'is_correct']]

    stu_question_accuracy_test_data = stu_question_accuracy_test_data[['question_id', 'user_id', 'subject_id1',
                                                                         'subject_id2', 'is_correct']]

    return stu_question_accuracy_train_data, stu_question_accuracy_valid_data, stu_question_accuracy_test_data


if __name__ == '__main__':
    # a = combine_data('../data/train_data.csv', '../data/question_meta.csv', '../data/train_data.csv', if_max=True)
    # print(a)
    question_subject = pd.read_csv('../data/question_meta.csv')
    # a,b,c,d = question_mapping(714,question_subject)
    # print(a,b,c,d)
    stu_question_accuracy_train_data = pd.read_csv('../data/train_data.csv')
    stu_question_accuracy_train_data = stu_question_accuracy_train_data.sort_values(by='question_id')
    list_ = list(map(lambda x: question_mapping(x, question_subject), stu_question_accuracy_train_data['question_id']))

    # print(list_)

    for num in range(2):
        stu_question_accuracy_train_data[f'subject_id{num + 1}'] = list(map(lambda x: x[num], list_))
    print(stu_question_accuracy_train_data)
    # list_0 = list(map(lambda x: x[0], list_))
    # list_1 = list(map(lambda x: x[1], list_))

    # print(list_0,list_1)
