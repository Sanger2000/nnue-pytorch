import chess
import torch

NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

# Factors are used by the trainer to share weights between common elements of
# the features.
FACTORS = {
  'kings': 64,
  'pieces': 640,
  #'relative_halfkp': 2250,
}
INPUTS = NUM_PLANES * NUM_SQ # 41024
FACTOR_INPUTS = sum(FACTORS.values())
INPUTS += FACTOR_INPUTS
NAME = 'HalfKP'

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def get_halfkp_indices(board: chess.Board):
  def piece_indices(turn):
    indices = torch.zeros(INPUTS - FACTOR_INPUTS)
    for sq, p in board.piece_map().items():
      if p.piece_type == chess.KING:
        continue
      indices[halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
    return indices
  return (piece_indices(chess.WHITE), piece_indices(chess.BLACK))
