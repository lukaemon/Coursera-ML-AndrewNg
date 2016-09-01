import helper.nn as nn
import numpy as np


def test_serial_deserial():
    t1, t2 = nn.load_weight('./helper/test/test_data/ex4weights.mat')

    o1, o2 = nn.deserialize(nn.serialize(t1, t2))

    assert np.mean(o1 == t1) == 1.0
    assert np.mean(o2 == t2) == 1.0
