import torch
from model import NNUE
import halfkp
import nnue_dataset
from data import fen_to_one_hot, output_stream
from nnue_bin_dataset import NNUEBinData
from fen_to_bin import fen_to_bitvec

b = '1sw3sw1ss/NE2se4/6sw1/1SE2sw3/3NE3ne/NE7/2NE4nw/EE3SW1NW1 W'
y = 0
out = fen_to_bitvec((b, y))

with open("test.bin", "wb") as f:
    for val in out:
        f.write(val)

val = NNUEBinData("test.bin")
print(len(val))

first = val.get_raw(0)
model = torch.load("out.pt")
print(first)

print(int(model(*first[:4])*600))
