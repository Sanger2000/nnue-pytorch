import torch
import chess
from model import NNUE
import halfkp
import nnue_dataset
from data import fen_to_one_hot, output_stream
from nnue_bin_dataset import NNUEBinData, ToTensor
from fen_to_bin import fen_to_bitvec

b = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2'
bd = chess.Board(fen=b)
y = 0
#out = fen_to_bitvec((b, y))
first = ToTensor()((bd, None, 0.5, 1))
'''
with open("test.bin", "wb") as f:
    for val in out:
        f.write(val)

val = NNUEBinData("test.bin")
print(len(val))

first = val.get_raw(0)
'''
checkpoint = torch.load("models/default/version_3/checkpoints/epoch=0-val_loss=0.05.ckpt")
model = NNUE()
print(first)
model.load_state_dict(checkpoint['state_dict'])
print(int(model(first[0].reshape(1, -1), first[1].reshape(1, -1), \
            first[2].reshape(1, -1), first[3].reshape(1, -1))*600))
