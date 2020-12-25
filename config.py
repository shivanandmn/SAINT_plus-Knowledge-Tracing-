import torch


class Config:
    device = torch.device("cuda")
    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 32
    TRAIN_FILE = "../input/riiid-test-answer-prediction/train.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000
