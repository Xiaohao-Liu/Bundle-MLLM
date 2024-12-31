import os
import random
import numpy as np
import scipy.sparse as sp
import json
import torch
import torch.nn.functional as F
from torch.backends import cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from main import MAP_LETTER


class BundleTrainDataset(Dataset):
    def __init__(self, conf, b_i_pairs, b_i_graph, features, num_bundles, b_i_for_neg_sample, b_b_for_neg_sample,
                 neg_sample=1):
        self.conf = conf
        self.b_i_pairs = b_i_pairs
        self.b_i_graph = b_i_graph
        self.bundles_map = np.argwhere(self.b_i_graph.sum(axis=1) > 0)[
                           :, 0].reshape(-1)
        self.num_bundles = num_bundles
        self.num_items = self.b_i_graph.shape[1]
        self.neg_sample = neg_sample
        self.features = features

        self.b_i_for_neg_sample = b_i_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample

        self.len_max = int(self.b_i_graph.sum(axis=1).max())

        self.num_cans = conf["num_cans"]

        if self.len_max > self.conf["num_token"]:
            self.len_max = self.conf["num_token"]
        self.few_shot = self.conf["few_shot"] if "few_shot" in self.conf else -1  # -1 means use all data to train

        print(f"Train: {self.len_max}")

    def __getitem__(self, index):

        b_idx = self.bundles_map[index]

        b_i_i = torch.from_numpy(
            self.b_i_graph[b_idx].toarray()).squeeze()

        indices_ori = torch.argwhere(b_i_i)[:, 0][:self.len_max + 1]  # truncation
        # select one as the positive
        perm_indices = torch.randperm(indices_ori.size(0))
        len_seq = indices_ori.shape[0]
        indices = indices_ori[perm_indices][:len_seq // 2]
        true_indice = indices_ori[perm_indices][-1]

        seq_b_i_i = F.pad(indices, (0, self.len_max - len(indices)), value=self.num_items)

        false_indices = torch.argwhere(b_i_i == 0)[:, 0].view(-1)
        false_indices = false_indices[np.random.randint(0, false_indices.shape[0], size=self.num_cans - 1)].view(-1)

        indices = torch.cat([torch.tensor([true_indice]), false_indices])
        shuffled_indices = indices[torch.randperm(self.num_cans)]

        true_idx = torch.argwhere(shuffled_indices == true_indice)[:, 0]

        return b_idx, seq_b_i_i, true_idx, shuffled_indices

    def __len__(self):
        if self.few_shot == -1:
            return len(self.bundles_map)

        return self.few_shot  # (one pos + one neg)


class BundleTestDataset(Dataset):
    def __init__(self, conf, b_i_pairs_i, b_i_graph_i, b_i_pairs_gt, b_i_graph_gt, num_bundles, num_items):
        self.b_i_pairs_i = b_i_pairs_i
        self.b_i_graph_i = b_i_graph_i

        self.b_i_pairs_gt = np.random.permutation(b_i_pairs_gt)  # b_i_pairs_gt
        self.b_i_graph_gt = b_i_graph_gt

        self.num_bundles = num_bundles
        self.num_items = num_items

        self.num_cans = conf["num_cans"]

        self.shift = 0

        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        if self.len_max > conf["num_token"]:
            self.len_max = conf["num_token"]

        self.toy_eval = conf["toy_eval"] if "toy_eval" in conf else -1
        print(f"Val/Test: {self.len_max}")

    def __getitem__(self, index):
        index = self.shift + index
        b_idx, true_indice = self.b_i_pairs_gt[index]

        b_i_i = torch.from_numpy(
            self.b_i_graph_i[b_idx].toarray()).squeeze()
        b_i_gt = torch.from_numpy(
            self.b_i_graph_gt[b_idx].toarray()).squeeze()

        false_indices = torch.argwhere((b_i_i + b_i_gt) == 0).view(-1)

        false_indices = false_indices[np.random.randint(0, false_indices.shape[0], size=self.num_cans - 1)].view(-1)

        indices = torch.cat([torch.tensor([true_indice]), false_indices])
        shuffled_indices = indices[torch.randperm(self.num_cans)]
        true_idx = torch.argwhere(shuffled_indices == true_indice)[:, 0]

        input_indices = torch.argwhere(b_i_i)[:, 0]
        if len(input_indices) > self.len_max:
            input_indices = input_indices[torch.randperm(len(input_indices))][:self.len_max]

        b_i_i = torch.zeros_like(b_i_i)
        b_i_i[input_indices] = 1
        seq_b_i_i = F.pad(
            input_indices, (0, self.len_max - len(input_indices)), value=self.num_items)

        return b_idx, seq_b_i_i, true_idx, shuffled_indices

    def __len__(self):
        if self.toy_eval == -1:
            return len(self.b_i_pairs_gt)
        else:
            return self.toy_eval


class BundleTestDataset2(Dataset):
    def __init__(self, conf, b_i_graph_i, data, num_bundles, num_items):

        self.data = data

        self.num_bundles = num_bundles
        self.num_items = num_items

        self.num_cans = conf["num_cans"]

        self.shift = 0

        self.len_max = int(b_i_graph_i.sum(axis=1).max())
        if self.len_max > conf["num_token"]:
            self.len_max = conf["num_token"]

        self.toy_eval = conf["toy_eval"] if "toy_eval" in conf else -1
        print(f"Val/Test: {self.len_max}")

    def __getitem__(self, index):

        index = self.shift + index
        b_idx, seq_b_i_i, true_idx, shuffled_indices = [torch.tensor(i).long() for i in self.data[index]]

        return b_idx, seq_b_i_i, true_idx, shuffled_indices

    def __len__(self):
        if self.toy_eval == -1:
            return len(self.data)
        else:
            return self.toy_eval


class Datasets():
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['data_path']
        self.name = conf['dataset']
        self.device = conf["device"]
        self.is_openai_embedding = conf["is_openai_embedding"] if "is_openai_embedding" in conf else False
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        u_i_pairs, u_i_graph = self.get_ui()

        b_i_pairs_train, b_i_graph_train = self.get_bi_train()
        b_i_pairs_val_i, b_i_graph_val_i, b_i_pairs_val_gt, b_i_graph_val_gt = self.get_bi(
            "valid")
        b_i_pairs_test_i, b_i_graph_test_i, b_i_pairs_test_gt, b_i_graph_test_gt = self.get_bi(
            "test")

        b_i_for_neg_sample, b_b_for_neg_sample = None, None

        b_i_pairs_seen, b_i_graph_seen = self.combine_graph(
            [b_i_pairs_train, b_i_pairs_val_i, b_i_pairs_test_i],
            shape=(self.num_bundles, self.num_items),
            tag="BI(seen)")
        self.graphs = [u_i_graph, b_i_graph_train, b_i_graph_seen]

        self.features = self.get_features()
        with open(os.path.join(self.path, self.name, 'item_info.json'), 'r') as f:
            self.item_info = json.loads(f.read())

        self.train_data = BundleTrainDataset(
            conf, b_i_pairs_train, b_i_graph_train, self.features, self.num_bundles, b_i_for_neg_sample,
            b_b_for_neg_sample, conf["neg_num"])

        b_i_val_cans = self.get_bi_cans(
            "valid")
        b_i_test_cans = self.get_bi_cans(
            "test")
        if b_i_test_cans == None:
            self.val_data = BundleTestDataset(conf, b_i_pairs_val_i, b_i_graph_val_i, b_i_pairs_val_gt,
                                              b_i_graph_val_gt,
                                              self.num_bundles, self.num_items)
            self.test_data = BundleTestDataset(conf, b_i_pairs_test_i, b_i_graph_test_i, b_i_pairs_test_gt,
                                               b_i_graph_test_gt,
                                               self.num_bundles, self.num_items)
        else:
            self.val_data = BundleTestDataset2(conf, b_i_graph_val_gt, b_i_val_cans,
                                               self.num_bundles, self.num_items)
            self.test_data = BundleTestDataset2(conf, b_i_graph_test_gt, b_i_test_cans,
                                                self.num_bundles, self.num_items)

        self.train_loader = DataLoader(
            self.train_data, batch_size=batch_size_train, shuffle=True, num_workers=10)
        self.val_loader = DataLoader(
            self.val_data, batch_size=batch_size_test, shuffle=True, num_workers=20)
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size_test, shuffle=True, num_workers=20)

    def combine_graph(self, pairs_list, shape, tag):
        pairs = np.concatenate(pairs_list, axis=0)
        indice = np.array(pairs, dtype=np.int32)
        values = np.ones(len(pairs), dtype=np.float32)
        graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=shape)
        return pairs, graph

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, 'count.json'), 'r') as f:
            self.stat = json.loads(f.read())
        return self.stat["#U"], self.stat["#B"], self.stat["#I"]

    def get_features(self):
        try:
            content_feature = torch.load(os.path.join(
                self.path, self.name, 'content_feature.pt'), map_location=self.device)
            if not self.is_openai_embedding:
                description_feature = torch.load(os.path.join(
                    self.path, self.name, 'description_feature.pt'), map_location=self.device)
            else:
                description_feature = torch.load(os.path.join(
                    self.path, self.name, 'openai_description_feature.pt'), map_location=self.device)
        except:
            print("[ERROR] no content_feature & description_feature")
            content_feature = description_feature = None

        cf_feature = torch.load(os.path.join(
            self.path, self.name, 'item_cf_feature.pt'), map_location=self.device)

        bi_feature = torch.load(os.path.join(
            self.path, self.name, f'{self.conf["dataset"]}_LightGCN_bi_feature.pt'), map_location=self.device)
        return (content_feature, description_feature, cf_feature, bi_feature)

    def get_ui(self):
        u_i_pairs = list2pairs(os.path.join(self.path, self.name, 'ui_full.txt'))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items))

        return u_i_pairs, u_i_graph

    def get_bi_train(self):

        b_i_pairs = list2pairs(os.path.join(self.path, self.name, 'bi_train.txt'))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items))

        return b_i_pairs, b_i_graph

    def get_bi(self, task):

        b_i_pairs_i = list2pairs(os.path.join(
            self.path, self.name, f'bi_{task}_input.txt'))
        b_i_pairs_gt = list2pairs(os.path.join(
            self.path, self.name, f'bi_{task}_gt.txt'))

        b_i_graph_i = pairs2csr(
            b_i_pairs_i, (self.num_bundles, self.num_items))
        b_i_graph_gt = pairs2csr(
            b_i_pairs_gt, (self.num_bundles, self.num_items))

        return b_i_pairs_i, b_i_graph_i, b_i_pairs_gt, b_i_graph_gt

    def get_bi_cans(self, task):
        try:
            path = os.path.join(
                self.path, self.name, f'bi_{task}_cans{self.conf["num_cans"]}.txt')

            data = []
            with open(path, "r") as f:
                data = [eval(i) for i in f]
            return data
        except:
            return None


def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape)


def list2pairs(file):
    pairs = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            l = [int(i) for i in line.split(", ")]
            b_id = l[0]
            for i_id in l[1:]:
                pairs.append([b_id, i_id])
    return np.array(pairs)


class DataCollator:
    def __init__(self):
        pass

    def __call__(self, samples):
        # list of dict, len(samples) = batch_size
        # return dict, each key has a list of values
        # import pdb;pdb.set_trace();
        # b_id, seq_b_i_i, true_position, candidates
        return {
            "gt_label": torch.stack([sample[2] for sample in samples]).squeeze(),
            "indices": torch.stack([sample[1] for sample in samples]),
            "candidates": torch.stack([sample[3] for sample in samples]),
        }


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def preprocess_logits_for_metrics(logits, gt_label):
    """
    Input:  logits: tensor [batch_size, seq_len, vocab_size]
            gt_label: list [batch_size * 1 or 0]
    """
    # take the -3 token as the prediction. Left Padding. [-100]+[3869, 29889, 2]
    add_eos_flag = 0
    probs = torch.argmax(logits[:, -3 - add_eos_flag], dim=-1)
    preds = []
    for i in probs.tolist():
        token_id = i
        pred = MAP_LETTER[token_id] if token_id in MAP_LETTER else -1
        preds.append(pred)
    return torch.tensor(preds), gt_label


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    count_hit = sum(pre[0] == pre[1])
    count_all = pre[0].shape[0]
    return {'hitrate': count_hit / count_all}
