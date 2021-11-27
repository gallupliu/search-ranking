HYS_CONFIG = {
    # label
    'columns': ['id', 'keyword', "item", 'title', 'brand', 'tag', 'volume', 'type', 'price', 'user_bert_emb',
                'item_bert_emb',
                'label'],
    'vocab_size': {
        'type': 3,

    },
    'vocab_file': './char.txt',
    'deep_emb_cols': ['type'],
    'deep_bucket_emb_cols': ['volume', 'price'],
    'wide_muti_hot_cols': ['type'],
    'wide_bucket_cols': ['volume', 'price'],
    'wide_cross_cols': [('type', 'volume'), ],
    'text_cols': ['keyword', "item"],
    'emb_cols': ['user_bert_emb', 'item_bert_emb'],
    'categorical_cols': ['type'],  # 类别型特征统一为string格式
    'numeric_cols': ['price'],#数值型，必须分桶，也就是对应cols 必须有bins
    'user_cols': [{'name': 'keyword', 'num': 5, 'embed_dim': 50}],
    'item_cols': [{'name': 'item', 'num': 15, 'embed_dim': 50},
                  {'name': 'type', 'num': 4, 'embed_dim': 2, 'vocab_list': ['0', '1', '2']},
                  {'name': 'volume', 'num': 1, 'embed_dim': 2},
                  {'name': 'price', 'num': 7, 'embed_dim': 2, 'bins': [0, 10, 20, 30, 40, 50]}
                  ]
}
