import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
import tensorflow.feature_column as fc
from transformers import AutoTokenizer, TFAutoModel
from models.keras.layers.transformer import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, user_num_layers, user_d_model, user_num_heads, item_num_layers, item_d_model, item_num_heads,
                 dff, vocab_size, maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()

        self.user_encoder = Encoder(num_layers=user_num_layers, d_model=user_d_model, num_heads=user_num_heads,
                                    dff=dff, input_vocab_size=vocab_size,
                                    maximum_position_encoding=maximum_position_encoding)

        self.item_encoder = Encoder(num_layers=item_num_layers, d_model=item_d_model, num_heads=item_num_heads,
                                    dff=dff, input_vocab_size=vocab_size,
                                    maximum_position_encoding=maximum_position_encoding)

    def call(self, user, item, enc_user_padding_mask,
             enc_item_padding_mask, training):
        user_enc_output = self.user_encoder(user, training, enc_user_padding_mask)  # (batch_size, inp_seq_len, d_model)
        item_enc_output = self.item_encoder(item, training, enc_item_padding_mask)  # (batch_size, inp_seq_len, d_model)
        return user_enc_output, item_enc_output


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = TFAutoModel.from_pretrained("bert-base-uncased")
# model = TFAutoModel.from_pretrained("/Users/gallup/data/bert/chinese_L-12_H-768_A-12", from_tf=True)
string_zh = '胃 康 宁 胶 囊'
inputs = tokenizer_zh(string_zh, max_length=10, padding="max_length", truncation=True)
tokenized_sequence_zh = tokenizer_zh.tokenize(string_zh)
# outputs = model(**inputs)
tokenized_sequence_decode_zh = tokenizer_zh.decode(inputs['input_ids'])
print('inputs:{0} type{1},{2}'.format(inputs, tokenized_sequence_zh, tokenized_sequence_decode_zh))
string_list = ['胃', '康', '宁', '胶', '囊']
inputs_list = tokenizer_zh(string_list, max_length=10, padding="max_length", truncation=True,is_split_into_words=True)
tokenized_sequence_decode_zh_list = tokenizer_zh.decode(inputs_list['input_ids'])
print('inputs_list:{0} list{1}'.format(inputs_list,  tokenized_sequence_decode_zh_list))
string_en = 'Hello world!'
model_inputs = tokenizer(string_en, max_length=10, padding="max_length", truncation=True)
tokenized_sequence_en = tokenizer.tokenize(string_en)
tokenized_sequence_decode = tokenizer.decode(model_inputs['input_ids'])
print('model_inputs:{0} type{1}'.format(model_inputs, type(model_inputs)))
print('inputs_ids:{0} type{1} {2}'.format(model_inputs['input_ids'], tokenized_sequence_en, tokenized_sequence_decode))

user_input = tf.random.uniform((64, 62))
item_input = tf.random.uniform((64, 26))

sample_transformer = Transformer(2, 64, 8, 2, 128, 8,
                                 512, 21128, 10000, rate=0.1)
user_out, item_out = sample_transformer(user_input, item_input, enc_user_padding_mask=None,
                                        enc_item_padding_mask=None, training=False, )

print(user_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
print(item_out.shape)
