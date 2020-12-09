import torch
import chess
from model import NNUE
import halfkp
import nnue_dataset
from data import fen_to_one_hot, output_stream
from nnue_bin_dataset import NNUEBinData, ToTensor
from fen_to_bin import fen_to_bitvec
import random

'''
b = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2'
bd = chess.Board(fen=b)
y = 0
#out = fen_to_bitvec((b, y))
first = ToTensor()((bd, None, 0.5, 1))
with open("test.bin", "wb") as f:
    output_stream(b, y, f)

val_filename = "test.bin"
val_infinite = nnue_dataset.SparseBatchDataset(halfkp.NAME, val_filename, 1, filtered=False)
val = nnue_dataset.FixedNumBatchesDataset(val_infinite, 100000)
new_train = nnue_dataset.FixedNumBatchesDataset(nnue_dataset.SparseBatchDataset(halfkp.NAME, "../train.bin", 1, num_workers=1, filtered=False), 1000)

first_1 = new_train[0]
first_2 = train[0]
train = NNUEBinData("../train.bin")

for i, val in enumerate(first_1[2]):
    if val != 0:
        print(i, val)

for i, val in enumerate(first_2[2]):
    if val != 0:
        print(i, val)

assert(first_1[2] == first_2[2])
'''
train = NNUEBinData("../valid2.bin")
idx = int(random.random()*len(train))

first = train.get_raw(idx)
checkpoint = torch.load("models/default/version_3/checkpoints/epoch=0-val_loss=0.05.ckpt")
model = NNUE()
print(first)
model.load_state_dict(checkpoint['state_dict'])


