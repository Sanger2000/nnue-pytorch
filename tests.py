import torch
from model import NNUE
import halfkp
import nnue_dataset
from data import fen_to_bytestring, output_stream
from nnue_bin_dataset import NNUEBinData
from fen_to_bin import fen_to_bitvec

b = [fen_to_bytestring('1sw3sw1ss/NE2se4/6sw1/1SE2sw3/3NE3ne/NE7/2NE4nw/EE3SW1NW1 W')]
y = [0]
#out = fen_to_bitvec((b, y))

with open("test.bin", "wb") as f:
    output_stream(b, y, f)

val_filename = "test.bin"
val_infinite = nnue_dataset.SparseBatchDataset(halfkp.NAME, val_filename, 1, filtered=False)
val = nnue_dataset.FixedNumBatchesDataset(val_infinite, 100000)
#val = NNUEBinData("test.bin")
print(len(val))

#first = val.get_raw(0)
first = val[0]
model = NNUE()
model2 = torch.load("out.pt")
print(first)

model.input.weight.data = model2.input.weight.data
model.input.bias.data = model2.input.bias.data

model.l1.weight.data = model2.l1.weight.data
model.l1.bias.data = model2.l1.bias.data

model.l2.weight.data = model2.l2.weight.data
model.l2.bias.data = model2.l2.bias.data

model.output.weight.data = model2.output.weight.data
model.output.bias.data = model2.output.bias.data

print(int(model(*first[:4])*600))
