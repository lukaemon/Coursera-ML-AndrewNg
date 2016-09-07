import pandas as pd
import scipy.io as sio


# support functions
def load_data(path):
    mat = sio.loadmat(path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')

    return data
