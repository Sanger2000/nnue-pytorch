import numpy as np
import time
from multiprocessing import Pool

pawn_orientations = {'ne': 0, 'nw': 1, 'sw': 2, 'se': 3}
king_orientations = {'ee': 0, 'nn': 1, 'ww': 2, 'ss': 3}

def encode_bit(data, pos, value):
    if np.uint16(value) != np.uint16(0):
        data[int(pos/8)] = data[int(pos/8)] | (1 << (pos & np.uint16(7)))

    return pos + 1

def encode_bits(data, pos, value, nbits):
    for i in range(0, nbits):
        pos = encode_bit(data, pos, np.uint16(value & np.uint16(1 << i)))

    return pos

def fen_to_bitvec(datapoint):
    fen_string, score = datapoint
    positions, color = fen_string.split()
    pos_dict = {}
    positions = positions.split('/')
    not_black = color != 'B'

    kings = [None for _ in range(2)]
    indices = []
    data = []

    loc = 0
    for i, pos in enumerate(positions):
        j = 0
        while (j < len(pos)):
            if pos[j].isnumeric():
                loc += int(pos[j])
                j += 1
            else:
                elem = pos[j:j+2].lower()
                is_white = (elem != pos[j:j+2])

                if elem in pawn_orientations:
                    pos_dict[loc] = (pawn_orientations[elem], np.uint16(not is_white))
                else:
                    kings[int(not is_white)] = loc
                loc += 1
                j += 2

    return encode_position(pos_dict, kings[0], kings[1], score, not_black)

def encode_position(pos_dict, white_king_pos, black_king_pos, score, is_white):
    data = np.zeros(40, dtype='uint8')
    pos = 0

    pos = encode_bit(data, pos, np.uint16(not is_white))
    pos = encode_bits(data, pos, np.uint16(white_king_pos), 6)
    pos = encode_bits(data, pos, np.uint16(black_king_pos), 6)

    for rank_ in range(8)[::-1]:
        for fil_ in range(8):
          i = rank_*8 + fil_
          if i == white_king_pos or i == black_king_pos:
              continue
          if i in pos_dict:
              piece_num, color = pos_dict[i]
              pos = encode_bit(data, pos, np.uint16(1))
              pos = encode_bits(data, pos, piece_num, 3)
              pos = encode_bit(data, pos, color)
          else:
              pos = encode_bit(data, pos, 0)

    pos = encode_bits(data, pos, np.int16(score), 16)

    return data

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def fens_to_bitvec(fens):
    return [fen_to_bitvec(fen) for fen in fens]
if __name__ == '__main__':
    lines = None
    with open("/Users/amansanger/Desktop/6.172/project4/project4/training/remote.fen", "r") as fen_file:
        lines = fen_file.read().strip().split('\n')


    fens = [lines[i] for i in range(0, len(lines), 2)]
    scores = [int(lines[i]) for i in range(1, len(lines), 2)]

    first_fens = fens[:10000]
    first_scores = scores[:10000]

    split_data = chunkIt(list(zip(first_fens, first_scores)), 8)


    with Pool(8) as p:
        processed_data = p.map(fens_to_bitvec, split_data)

    processed_data = np.concatenate(processed_data).flatten()
    with open("train_data.bin", "wb") as train_file:
        for elem in processed_data:
            train_file.write(elem)

    # VALID DATA NOW:

    second_fens = fens[-10000:]
    second_scores = scores[-10000:]

    split_data = chunkIt(list(zip(second_fens, second_scores)), 8)

    with Pool(8) as p:
        processed_data = p.map(fens_to_bitvec, split_data)

    processed_data = np.concatenate(processed_data).flatten()
    with open("valid_data.bin", "wb") as valid_file:
        for elem in processed_data:
            valid_file.write(elem)

