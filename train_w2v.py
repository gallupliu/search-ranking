import os
import sys
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(l'
                           'evelname)s: %(message)s', level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型
inp = '~/Downloads/wiki.zh.simp.seg.txt'
outp1 = './data/model/Word2Vec/wiki.zh.text.model'
outp2 = './data/model/Word2Vec/wiki.zh.text.vector'

# 训练skip-gram 模型
model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)