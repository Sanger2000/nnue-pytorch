import chess
import halfkp
import mmap
import random
import os
import torch
import torch.nn.functional as F
import numpy as np

PACKED_SFEN_VALUE_BYTES = 40

HUFFMAN_MAP = {0b000 : chess.PAWN, 0b001 : chess.KNIGHT, 0b010 : chess.BISHOP, 0b011 : chess.ROOK, 0b100: chess.QUEEN}

reverse_map = {'p': 'ne', 'n': 'nw', 'b': 'sw', 'r': 'se', 'k': 'k'}
king_map = ['nn', 'ee', 'ss', 'ww']

def convert_fen(fen, white_ori, black_ori):
  out = ""
  i = 0
  while i < len(fen):
      if fen[i].isnumeric() or fen[i] == "/":
          out += fen[i]
      else:
          val = fen[i].lower()
          isupper = (val != fen[i])

          if val == 'k':
              if isupper:
                  out += king_map[white_ori].upper()
              else:
                  out += king_map[black_ori]

          else:
              if isupper:
                  out += reverse_map[val].upper()
              else:
                  out += reverse_map[val]

      i += 1

  return out

def twos(v, w):
  return v - int((v << 1) & 2**w)

class BitReader():
  def __init__(self, bytes, at):
    self.bytes = bytes
    self.seek(at)

  def readBits(self, n):
    r = self.bits & ((1 << n) - 1)
    self.bits >>= n
    self.position -= n
    return r

  def refill(self):
    while self.position <= 24:
      self.bits |= self.bytes[self.at] << self.position
      self.position += 8
      self.at += 1

  def seek(self, at):
    self.at = at
    self.bits = 0
    self.position = 0
    self.refill()

def is_quiet(board, from_, to_):
  for mv in board.legal_moves:
    if mv.from_square == from_ and mv.to_square == to_:
      return not board.is_capture(mv)
  return False

class ToTensor(object):
  def __call__(self, sample):
    bd, _, outcome, score = sample
    us = torch.tensor([bd.turn])
    them = torch.tensor([not bd.turn])
    outcome = torch.tensor([outcome])
    score = torch.tensor([score])
    white, black = halfkp.get_halfkp_indices(bd)
    return us.float(), them.float(), white.float(), black.float(), outcome.float(), score.float()

class RandomFlip(object):
  def __call__(self, sample):
    bd, move, outcome, score = sample
    mirror = random.choice([False, True])
    if mirror:
      bd = bd.mirror()
    return bd, move, outcome, score

class LeiserNNUEBinData(torch.utils.data.Dataset):
  def __init__(self, filename):
    super(NNUEBinData, self).__init__()
    self.filename = filename
    self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
    self.file = None

  def __len__(self):
    return self.len

  def get_raw(self, idx):
    if self.file is None:
      self.file = open(self.filename, 'r+b')
      self.bytes = mmap.mmap(self.file.fileno(), 0)

    base = PACKED_SFEN_VALUE_BYTES * idx
    br = BitReader(self.bytes, base)


    bd = chess.Board(fen=None)
    turn = not br.readBits(1)
    white_king_sq = br.readBits(6)
    white_king_ori = br.readBits(2)

    black_king_sq = br.readBits(6)
    black_king_ori = br.readBits(2)

    bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
    
    assert(black_king_sq != white_king_sq)

    white_offset = white_king_sq * white_king_ori * halfkp.NUM_LEISER_PLANES
    black_offset = black_king_sq * black_king_ori * halfkp.NUM_LEISER_PLANES

    # Output sparse vectors:
    whites = torch.zeros(halfkp.LEISER_INPUTS, dtype=torch.Float)
    blacks = torch.zeros(halfkp.LEISER_INPUTS, dtype=torch.Float)
    
    for rank_ in range(8)[::-1]:
      br.refill()
      for file_ in range(8):
        i = chess.square(file_, rank_)
        assert(i == rank_*8 + file_)
        if white_king_sq == i or black_king_sq == i:
          continue
        if br.readBits(1):
          ori = br.readBits(2)
          color = br.readBits(1)

          whites[white_offset + i*8 + color*4 + ori] = 1.0
          blacks[black_offset + i*8 + color*4 + ori] = 1.0

          br.refill()
    
    br.seek(base + 32)
    next_val = br.readBits(16)
    score = twos(next_val, 16)
    
    move = br.readBits(16)
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6
    
    br.refill()
    ply = br.readBits(16)
    bd.fullmove_number = ply // 2


    move = None
    
    # 1, 0, -1
    game_result = br.readBits(8)
    outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]

    # Tensors that can be fed into neural network
    us = torch.tensor([turn], dtype=torch.Float)
    them = torch.tensor([not turn], dtype=torch.Float)
    outcome = torch.tensor([outcome], dtype=torch.Float)
    score = torch.tensor([score], dtype=torch.Float)

    return us, them, white, black, outcome, score

  def __getitem__(self, idx):
    return self.get_raw(idx)

  # Allows this class to be pickled (otherwise you will get file handle errors).
  def __getstate__(self):
    state = self.__dict__.copy()
    state['file'] = None
    state.pop('bytes', None)
    return state

class NNUEBinData(torch.utils.data.Dataset):
  def __init__(self, filename, transform=ToTensor()):
    super(NNUEBinData, self).__init__()
    self.filename = filename
    self.len = os.path.getsize(filename) // PACKED_SFEN_VALUE_BYTES
    print(self.len)
    self.transform = transform
    self.file = None

  def __len__(self):
    return self.len

  def get_raw(self, idx):
    if self.file is None:
      self.file = open(self.filename, 'r+b')
      self.bytes = mmap.mmap(self.file.fileno(), 0)

    base = PACKED_SFEN_VALUE_BYTES * idx
    br = BitReader(self.bytes, base)

    bd = chess.Board(fen=None)
    bd.turn = not br.readBits(1)
    white_king_sq = br.readBits(6)
    black_king_sq = br.readBits(6)
    bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
    
    assert(black_king_sq != white_king_sq)
    
    for rank_ in range(8)[::-1]:
      br.refill()
      for file_ in range(8):
        i = chess.square(file_, rank_)
        assert(i == rank_*8 + file_)
        if white_king_sq == i or black_king_sq == i:
          continue
        if br.readBits(1):
          assert(bd.piece_at(i) == None)
          piece_index = br.readBits(3)
          piece = HUFFMAN_MAP[piece_index]
          color = br.readBits(1)
          bd.set_piece_at(i, chess.Piece(piece, not color))
          br.refill()
    
    br.seek(base + 32)
    next_val = br.readBits(16)
    score = twos(next_val, 16)
    
    move = br.readBits(16)
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6
    
    br.refill()
    ply = br.readBits(16)
    #bd.fullmove_number = ply // 2

    white_ori = ply & 3
    black_ori = (ply >> 2) & 3

    move = chess.Move(from_square=chess.SQUARES[from_], to_square=chess.SQUARES[to_])
    
    # 1, 0, -1
    game_result = br.readBits(8)
    outcome = {1: 1.0, 0: 0.5, 255: 0.0}[game_result]
    outs = bd.fen().split()[:2]
    print(convert_fen(outs[0], white_ori, black_ori) + " " + outs[1].upper())
    print("outcome: ", outcome)
    print("score: ", score)
    print("oris : ", white_ori, black_ori)
    return bd, move, outcome, score

  def __getitem__(self, idx):
    item = self.get_raw(idx)
    return self.transform(item)

  # Allows this class to be pickled (otherwise you will get file handle errors).
  def __getstate__(self):
    state = self.__dict__.copy()
    state['file'] = None
    state.pop('bytes', None)
    return state
