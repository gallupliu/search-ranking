import numpy as np
import din_estimator
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():
    feature_columns = [SparseFeat('driver_age', 7, embedding_dim=32),
                       SparseFeat('pax_age', 7, embedding_dim=32),
                       SparseFeat('des_id', 10000, embedding_dim=32),
                       SparseFeat('price_id', 20, embedding_dim=32)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_price_id', vocabulary_size=5, embedding_dim=32), maxlen=3),
        VarLenSparseFeat(SparseFeat('hist_des_id', vocabulary_size=5, embedding_dim=32), maxlen=3)]
    # Notice: History behavior sequence feature name must start with "hist_".
    behavior_feature_list = ["price_id", "des_id"]
    driver_age = np.array([0, 1, 2])
    pax_age = np.array([0, 1, 0])
    pax_des = np.array([1, 2, 3])  # 0 is mask value
    pax_price = np.array([1, 2, 2])  # 0 is mask value

    hist_price_seq = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 0]])
    hist_des_seq = np.array([[1, 2, 2], [2, 2, 1], [1, 2, 0]])

    feature_dict = {'driver_age': driver_age, 'pax_age': pax_age, 'des_id': pax_des, 'price_id': pax_price,
                    'hist_price_id': hist_price_seq, 'hist_des_id': hist_des_seq}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    print(x)
    print('\n')
    print(y)
    print('\n')
    print(feature_columns)
    print('\n')
    print(behavior_feature_list)
    print('\n')
    model = DIN(feature_columns, behavior_feature_list)
    # model = BST(feature_columns, behavior_feature_list,att_head_num=4)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
