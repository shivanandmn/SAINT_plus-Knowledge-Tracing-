from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc


class DKTDataset(Dataset):
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        for id in self.samples.index:
            exe_ids, answers, ela_time, categories = self.samples[id]
            if len(exe_ids) > max_seq:
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append(
                        (exe_ids[l:l+max_seq], answers[l:l+max_seq], ela_time[l:l+max_seq], categories[l:l+max_seq]))
            elif len(exe_ids) < self.max_seq and len(exe_ids) > 50:
                self.data.append((exe_ids, answers, ela_time, categories))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, ela_time, exe_category = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros(self.max_seq, dtype=int)
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            elapsed_time[-seq_len:] = ela_time
            exe_cat[-seq_len:] = exe_category
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsed_time, 0, 0)
        input_rtime = np.delete(input_rtime, -1)

        input = {"input_ids": exe_ids, "input_rtime": input_rtime.astype(
            np.int), "input_cat": exe_cat}
        return input, ans


def get_dataloaders():
    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'int8', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    print("loading csv.....")
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[
                           1, 2, 3, 4, 5, 7, 8], dtype=dtypes, nrows=90e6)
    print("shape of dataframe :", train_df.shape)

    train_df = train_df[train_df.content_type_id == 0]
    train_df.prior_question_elapsed_time.fillna(0, inplace=True)
    train_df.prior_question_elapsed_time /= 1000
    # train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(
        np.int)

    train_df = train_df.sort_values(
        ["timestamp"], ascending=True).reset_index(drop=True)
    n_skills = train_df.content_id.nunique()
    print("no. of skills :", n_skills)
    print("shape after exlusion:", train_df.shape)

    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id", "content_id", "answered_correctly", "prior_question_elapsed_time", "task_container_id"]]\
        .groupby("user_id")\
        .apply(lambda r: (r.content_id.values, r.answered_correctly.values,
                          r.prior_question_elapsed_time.values, r.task_container_id.values))
    del train_df
    gc.collect()
    print("splitting")
    train, val = train_test_split(group, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)
    train_dataset = DKTDataset(train, max_seq=Config.MAX_SEQ)
    val_dataset = DKTDataset(val, max_seq=Config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False)
    del train_dataset, val_dataset
    gc.collect()
    return train_loader, val_loader
