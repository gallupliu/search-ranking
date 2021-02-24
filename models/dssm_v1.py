from collections import namedtuple, OrderedDict
import tensorflow as tf

SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'share_embed','embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed','reduce_type','dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'share_embed', 'weight_name', 'embed_dim','maxlen', 'dtype'])



