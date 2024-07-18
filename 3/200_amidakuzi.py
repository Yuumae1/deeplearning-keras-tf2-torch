'''
あみだくじをシグモイド関数で実装する
'''
import numpy as np

class LogisticRegression(object):
    '''
    ロジスティック回帰
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0.

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        y = sigmoid(np.matmul(x, self.w) + self.b)
        return y

    def compute_gradients(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = np.matmul(x.T, delta)
        db = np.matmul(np.ones(x.shape[0]), delta)

        return dw, db

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    np.random.seed(123)

    '''
    1. データの準備
    '''
    # あみだくじ設計図（入力, 出力）= (4, 4)
    x = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    t = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    
    '''
    2. モデルの構築
    '''
    model = LogisticRegression(input_dim=4)
    
    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()
    
    def train_step(x, t):
        dw, db = model.compute_gradients(x, t)
        model.w = model.w - 0.1 * dw
        model.b = model.b - 0.1 * db
        loss = compute_loss(t, model(x))
        return loss
    
    for epoch in range(1, 10000):
        loss = train_step(x, t)
        if epoch % 100 == 0:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch,
                loss
            ))
    
    '''
    4. モデルの評価
    '''
    for input in x:
        print(model(input))
    print(t)
        
    
    