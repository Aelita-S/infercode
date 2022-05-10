import csv
import os
import pickle as pkl
from typing import Collection

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, confusion_matrix, recall_score, \
    roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from infercode.client.infercode_client import InferCodeClient
from infercode.settings import DATASET_DIR, SAVED_MODEL_PATH

csv.field_size_limit(500 * 1024 * 1024)


class BaseHandler:
    def __init__(self):
        self.y_true = None
        self.y_score = None
        self.y_pred = None


class OJCloneHandler(BaseHandler):
    programs_path = DATASET_DIR / 'astnn' / 'c' / 'programs.pkl'
    clone_pairs_path = DATASET_DIR / 'astnn' / 'c' / 'oj_clone_ids.pkl'

    def read_programs(self) -> tuple:
        """
        Reads the ASTNN's OJClone programs from the given path.
        :return: List of ASTNN and OJClone programs.
        """
        with open(self.programs_path, 'rb') as f:
            programs = pkl.load(f)
        code_list = programs[:, 1]
        return code_list

    def read_clone_pairs(self):
        with open(self.clone_pairs_path, 'rb') as f:
            clone_pairs = pkl.load(f)
        return clone_pairs

    def evaluate(self, y_true, y_score, threshold=0.97):
        print(f'confusion matrix:\n{confusion_matrix(y_true, y_score > threshold)}\n')
        print(f'classification_report:\n{classification_report(y_true, y_score > threshold, digits=3)}\n')


class BCBHandler(BaseHandler):
    bcb_path = DATASET_DIR / 'astnn' / 'java' / 'bcb_funcs_all.tsv'
    clone_pairs_path = DATASET_DIR / 'astnn' / 'java' / 'bcb_pair_ids.pkl'

    def __init__(self):
        super().__init__()
        self.indexes = {}
        self.programs = None
        self.clone_pairs = None

    def read_programs(self):
        with open(self.bcb_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = [line for line in tqdm(reader, desc='Reading BCB dataset: ')]
        funcs = []
        i = 0
        for line in tqdm(lines, desc="Reading BCB functions: "):
            program_lines = line[1].split('\n')
            if len(program_lines) > 500:
                continue
            if max([len(line) for line in program_lines]) > 1000:
                continue
            self.indexes[int(line[0])] = i
            i += 1
            funcs.append(line[1])
        self.programs = funcs
        return funcs

    def read_clone_pairs(self):
        with open(self.clone_pairs_path, 'rb') as f:
            clone_pairs = pkl.load(f)
        # ID index conversion
        clone_pairs[:, 0] = [self.indexes[id] if id in self.indexes else 0 for id in clone_pairs[:, 0]]
        clone_pairs[:, 1] = [self.indexes[id] if id in self.indexes else 0 for id in clone_pairs[:, 1]]
        # filter exculded IDs
        clone_pairs = clone_pairs[clone_pairs[:, 0] != 0]
        clone_pairs = clone_pairs[clone_pairs[:, 1] != 0]
        # clone_pairs[:, -1] = np.where(clone_pairs[:, -1] >= 1, 1, 0)
        self.clone_pairs = clone_pairs
        return clone_pairs

    def evaluate(self, y_true, y_score, threshold=None):
        y_true_all = np.where(y_true >= 1, 1, 0)

        fpr_all, tpr_all, threshold_all = roc_curve(y_true_all, y_score)
        optimal_threshold, _ = get_optimal_cutoff(tpr_all, fpr_all, threshold_all)
        y_pred = y_score > optimal_threshold
        print(f'optimal threshold: {optimal_threshold}')
        print(f'confusion matrix:\n{confusion_matrix(y_true_all, y_pred)}\n')
        print(f'classification_report:\n{classification_report(y_true_all, y_pred, digits=3)}\n')

        for label in range(1, 6):
            y_true_label = y_true[y_true == label]
            y_true_label = np.where(y_true_label >= 1, 1, 0)
            y_score_label = y_score[y_true == label]
            y_pred_label = y_score_label > optimal_threshold
            recall = recall_score(y_true_label, y_pred_label)
            print(f'label {label}: y_true num: {len(y_true_label)} '
                  f'y_pred_pos num: {np.count_nonzero(y_pred_label == True)} '
                  f'recall: {recall:.3f}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        roc_auc = auc(fpr_all, tpr_all)
        plt.plot(fpr_all, tpr_all, label=f'ROC (area = {roc_auc:.3f})', lw=2)
        plt.legend(loc="lower right")
        plt.show()


def predict(language, code_snippets: Collection[str]) -> np.ndarray:
    infercode = InferCodeClient(model_path=SAVED_MODEL_PATH)
    code_vectors = [infercode.encode([code_snippet], language) for code_snippet in tqdm(code_snippets)]
    return np.concatenate(code_vectors)


def get_y(code_vectors, clone_pairs):
    y_true = clone_pairs[:, -1].astype(int)
    num_pairs = len(y_true)
    indexes_x = list(clone_pairs[:, 0])
    indexes_y = list(clone_pairs[:, 1])
    y_score = np.array([cosine_similarity([code_vectors[x]], [code_vectors[y]])[0][0] for x, y in
                        tqdm(zip(indexes_x, indexes_y), total=num_pairs)])
    return y_true, y_score


def show_roc_curve(fpr, tpr, type_note=""):
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'{type_note} ROC (area = {roc_auc:.3f})', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def get_optimal_cutoff(tpr, fpr, threshold):
    y = tpr - fpr
    youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[youden_index]
    point = [fpr[youden_index], tpr[youden_index]]
    return optimal_threshold, point


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    language = 'java'

    # handler = OJCloneHandler()
    # code_list = handler.read_programs()
    # code_vectors = predict('c', code_list)
    # with open(DATASET_DIR / 'astnn' / 'c' / 'code_vectors.pkl', 'wb') as f:
    #     pkl.dump(code_vectors, f)
    with open(DATASET_DIR / 'astnn' / language / 'code_vectors.pkl', 'rb') as f:
        code_vectors = pkl.load(f)
    handler = BCBHandler()
    code_list = handler.read_programs()
    clone_pairs = handler.read_clone_pairs()

    # code_vectors = predict(language, code_list)
    # with open(DATASET_DIR / 'astnn' / 'java' / 'code_vectors.pkl', 'wb') as f:
    #     pkl.dump(code_vectors, f)
    y_true, y_score = get_y(code_vectors, clone_pairs)
    handler.evaluate(y_true, y_score)
