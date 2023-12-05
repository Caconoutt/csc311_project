import csv
import os
import time
import numpy as np
from torch import tensor, Tensor

from typing import Dict


def load_train_question_correct_prop(path='../data/train_data.csv') -> np.ndarray:
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    correct = np.zeros((1774, ))
    total = np.zeros((1774, ))

    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            try:
                question_id = int(row[0])
                is_correct = int(row[2])
                correct[question_id] += is_correct
                total[question_id] += 1
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return correct / total

def load_subjects(path='../data/question_meta.csv') -> Dict[int, Tensor]:
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    subjects = {}
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            try:
                question_id = int(row[0])
                question_subjects = tensor(eval(row[1]))
                subjects[question_id] = question_subjects
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return subjects


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


def load_train_csv(root_dir="../data"):
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


def load_valid_csv(root_dir="../data"):
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


def load_public_test_csv(root_dir="../data"):
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


def main() -> None:
    question_correct_prop = load_train_question_correct_prop()
    subjects = load_subjects()


if __name__ == '__main__':
    main()