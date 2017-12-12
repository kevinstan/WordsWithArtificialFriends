import numpy as np

from chainer import Chain
import chainer.functions as F
import chainer.links as L

class charRNN(Chain):
    """Recurrent neural network model with 3-hidden LSTM layers:
        n_vocab : n_units : n_units : n_units : n_vocab
    """

    def __init__(self, n_vocab, n_units, train=True):
        super(charRNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))

        return y
