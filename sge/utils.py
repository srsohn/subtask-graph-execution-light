from enum import Enum
import numpy as np
from collections import defaultdict

MAX_DIST = 2**14-2

OID_TO_IID = lambda x: x+3

class KEY(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,
    PICKUP = 4,
    TRANSFORM = 5,
    USE_1 = 5,
    USE_2 = 6,
    USE_3 = 7,
    USE_4 = 8,
    USE_5 = 9,
    QUIT = 'q'


WHITE = (255, 255, 255)
LIGHT = (196, 196, 196)
GREEN = (80, 160, 0)
DARK = (128, 128, 128)
DARK_RED = (139, 0, 0)
BLACK = (0, 0, 0)

MOVE_ACTS = {KEY.UP, KEY.DOWN, KEY.LEFT, KEY.RIGHT}

EMPTY = -1
AGENT = 0
BLOCK = 1
WATER = 2
OBJ_BIAS = 3

TYPE_PICKUP = 0
TYPE_TRANSFORM = 1


def get_id_from_ind_multihot(indexed_tensor, mapping, max_dim):
    if type(mapping) == dict:
        mapping_ = np.zeros(max(mapping.keys())+1, dtype=np.long)
        for k, v in mapping.items():
            mapping_[k] = v
        mapping = mapping_
    if indexed_tensor.ndim == 2:
        nbatch = indexed_tensor.shape[0]
        out = np.zeros(nbatch, max_dim).astype(indexed_tensor.dtype)
        np.add.at(out, mapping.ravel(), indexed_tensor.ravel())
    else:
        out = np.zeros(max_dim).astype(indexed_tensor.dtype)
        np.add.at(out, mapping.ravel(), indexed_tensor.ravel())

    return out

class TimeProfiler(object):
  """Maintain moving statistics of time profile."""
  def __init__(self, prefix=''):
    self._prefix = 'time_profile/' + prefix
    self._buffer_dict = defaultdict(float)
    self._buffer_count = 0.
    self._safe_lock = defaultdict(bool)
    #
    self._prev_time_stamp = None

  def stamp(self, time_stamp, name=None):
    if name is not None:
      if self._prev_time_stamp is None:
        print("Error in Timeprofiler!!! You should call stamp() without 'name' argument after 'period_over()!!")
        assert False
      self._buffer_dict[name] += time_stamp - self._prev_time_stamp
    '''if self._prev_time_stamp is None: pass
    else:
      print(name, time_stamp - self._prev_time_stamp)'''
    self._prev_time_stamp = time_stamp
  
  def period_over(self):
    self._prev_time_stamp = None
    self._buffer_count += 1.
  
  def print(self):
    if self._prev_time_stamp is not None:
      print("Error in Timeprofiler!!! You should finish profiling full period before logging!")
      assert False
    for k, v in self._buffer_dict.items():
      print('%s = %.2f (sec)'%(self._prefix + k, v / self._buffer_count))
  
  def log_summary(self, summary_logger):
    if self._prev_time_stamp is not None:
      print("Error in Timeprofiler!!! You should finish profiling full period before logging!")
      assert False
    for k, v in self._buffer_dict.items():
      summary_logger.logkv(self._prefix + k, v / self._buffer_count)