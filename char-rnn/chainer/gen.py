import argparse
import sys
import six
import _pickle
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import serializers

from char_rnn import charRNN

def main():
    parser = argparse.ArgumentParser()

    ## Global config
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                       help= 'GPU id (negative value indicates CPU)')

    ## Input/output config
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Trained model result')
    parser.add_argument('--vocab', '-v', type=str, required=True,
                        help='Binary vocabulary object')
    parser.add_argument('--pretext', '-t', type=str, default='',
                        help='Pre-text for prediction/generation')
    parser.add_argument('--length', '-l', type=int, default=2000,
                        help='Length of generating text')

    ## Model config
    # n_units should be the same with n_units in train.py
    parser.add_argument('--n_units', '-n', type=int, default=128,
                        help='Number of LSTM unit in each layer')

    args = parser.parse_args()

    # Matrices processing is for GPU if specified
    xp = cuda.cupy if args.gpu >= 0 else np

    # Load vocabulary
    vocab = _pickle.load(open(args.vocab, 'rb'))
    # switch to index by number
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # Load model
    lm = charRNN(len(vocab), args.n_units, train=False)
    model = L.Classifier(lm)
    serializers.load_npz(args.model, model)

    # GPU-ize
    if args.gpu >= 0:
        cuda.get_device(0).use()
        model.to_gpu()

    # Reset state for predicting (not training anymore)
    model.predictor.reset_state()

    # Initial state
    prev_char = chainer.Variable(xp.array([0], dtype=xp.int32))

    # Load in pre-text
    pretext = args.pretext
    if isinstance(pretext, six.binary_type):
        pretext = pretext.decode('utf-8')

    if len(pretext) > 0:
        for ch in pretext:
             # First print pre-chars
            sys.stdout.write(ch)
            prev_char = chainer.Variable(xp.array([vocab[ch]], dtype=xp.int32))

            # Calculate probability
            prob = F.softmax(model.predictor(prev_char))


    for i in six.moves.range(args.length):
        # Predict
        prob = F.softmax(model.predictor(prev_char))
        # random choice
        probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
        prediction = ivocab[index]

        # Now print the predicted result
        sys.stdout.write(prediction)

        # Prepare for new predict (current char to prev_char)
        prev_char = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
