import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

#pawn_orientations = {'ne': 0b1000, 'nw': 0b1100, 'sw': 0b1010, 'se': 0b1110}
pawn_orientations = {'ne': 0b1000, 'nw': 0b1100, 'sw': 0b1010, 'se': 0b1110}
#                      pawn         knight        bishop        rook
king_orientations = {'ee': 0, 'nn': 1, 'ww': 2, 'ss': 3}

class PrepareData(Dataset):
    '''
    Dataset class for pytorch easy batching/shuffling
    '''

    def __init__(self, X, y):
        self.X = torch.zeros(len(X), 64*16).float()
        for idx, indices in enumerate(X):
            for index in indices:
                self.X[idx, index] = 1
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def bitstring_to_bytes(ns):
    assert(len(ns) == 32*8)
    s = ""
    for i in range(0, 32*8, 8):
        s += ns[i:i+8][::-1]

    #print(s[1:7], s[7:13])

    v = int(s, 2)
    b = bytearray()
    for _ in range(len(s)//8):
        b.append(v & 0xff)
        v >>= 8
    b = b[::-1]
    #return bytes(b[::-1])
    #print(len(s), len(b))
    assert(len(bytes(b)) == 32)
    assert(len(b) == 32)
    #return bytes(b[::-1])
    return bytes(b)
    '''rev = []
    for i in range(0, 32, 8):
        rev.extend(b[i:i+8][::-1])
    assert(len(rev) == 32)
    assert(len(bytes(rev)) == 32)
    return bytes(rev)
    '''

def fen_to_one_hot(fen_string):
    '''
    Converts a fen string to sparse storage of one-hot encoded vector
    If the player is black, then the black pieces are encoded as the
    lower index in one-hot. If the player is white, the opposite is true
    '''
    try:
        positions, color = fen_string.split()
        positions = positions.split('/')
        not_white = color != 'W'

        kings = [None for _ in range(2)]
        indices = []
        data = []

        loc = 0
        bin_val = ''
        bits = 1
        arr = []
        for i, pos in enumerate(positions):
            j = 0
            while (j < len(pos)):
                if pos[j].isnumeric():
                    loc += int(pos[j])

                    bits += int(pos[j])
                    bin_val += '0'*int(pos[j])

                    j += 1
                else:
                    #index = loc*8
                    index = loc

                    elem = pos[j:j+2].lower()
                    is_black = (elem == pos[j:j+2])
                    #index += 4*(is_black^not_white)

                    if elem in pawn_orientations:
                        data.append(1)

                        bits += 5
                        bin_val += "{0:05b}".format((0b00001 if is_black else 0) +  (pawn_orientations[elem] << 1))
                    else:
                        #kings[int(not is_black)] = loc*4 + king_orientations[elem]
                        kings[int(is_black)] = (loc, king_orientations[elem])
                    loc += 1
                    j += 2

        # If white, the white king must appear in the top half of data
        #if color == 'W':
        #    kings[0], kings[1] = kings[1], kings[0]

        cap = 32*8-12
        assert(bits <= cap)
        s = ("1" if not_white else "0") + "{0:06b}".format(kings[0][0]) + "{0:06b}".format(kings[1][0]) + bin_val + '0'*(cap-bits)

        if len(s) != 32*8:
            print(fen_string)
            return None

        return bitstring_to_bytes(s)

    except:
        print(fen_string)
        return None


def preprocess_scores(scores):
    # We trim outliers to be 1000 and -1000, then apply the sqrt
    # to get uniformity in data

    #y = np.array([int(score) for score in scores])
    y = [int(score) for score in scores]
    #y = [2 for score in scores]
    '''y = np.clip(y, -1000, 1000)
    y = np.sign(y)*np.sqrt(abs(y))

    y -= np.min(y)
    y /= np.max(y)
    y *= 127
    '''

    #return torch.tensor(y)
    return y

def data_loaders(X, y, batch_size=32):
    state = np.random.get_state()
    np.random.shuffle(y)

    np.random.set_state(state)
    np.random.shuffle(X)

    x_train = X[:20000]
    y_train = y[:20000]

    x_valid = X[20000:]
    y_valid = y[20000:]


    train_data = PrepareData(X=x_train, y=y_train)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_data = PrepareData(X=x_valid, y=y_valid)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_data, valid_data




def output_stream(x_set, y_set, f):
    for i in range(0, len(x_set), 1000):
        print(i)
        b = bytes()
        for j in range(i, min(i+1000, len(x_set))):
            x = x_set[j]
            if x is None:
                continue
            y = y_set[j]
            b += x + (y).to_bytes(2, byteorder='little', signed=True) + (0).to_bytes(6, byteorder='little')
        f.write(b)
        

def get_data(filename="remote.fen", batch_size=32):
    f = open(filename, 'r')
    lines = f.read().strip().split('\n')
    f.close()

    inc = 100000
    with open('out.bin', 'wb') as fout:
        for i in range(0, len(lines), inc):
            print(i)
            cap = min(i+inc, len(lines))
            fens = [lines[j] for j in range(i, cap, 2)]
            scores = [lines[j] for j in range(i+1, cap, 2)]

            X = [fen_to_one_hot(fen) for fen in fens]
            y = preprocess_scores(scores)

            state = np.random.get_state()
            np.random.shuffle(y)

            np.random.set_state(state)
            np.random.shuffle(X)

            output_stream(X, y, fout)

