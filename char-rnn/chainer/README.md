# Char-rnn generation (Chainer ver.)

Character-level language model implementation of Karpathy's char-rnn(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
with Long Short Term Memory (LSTM) shell.


## Requirement

+ Python (>= 3.4.0) recommended.


## Set-up

+ Install required packages:

  + Traditional way:

  ```
  $ pip3 install -r requirements.txt
  ```

  + `virtualenv` users:

  ```
  $ virtualenv -p /usr/bin/python3 venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt
  ```

## Demo

**NOTE**: All demo default with CPU, if you want to train/generate with gpu just parse argument: `--gpu [id]`.
For example, train/generate with first GPU: `--gpu 0`.

+ [Training](#training) (*Skip if you want to see directly generating process with pre-trained model*)

```
$ python train.py --data data/harry.txt
```

An example input `harry.txt` already supported with text format of J.K.Rowling's `Harry Potter and The Order of The Phoenix`


+ [Generating](#generating)

```
$ python gen.py --model result/harry_model \
  --vocab result/harry.bin \
  --pretext 'Harry Potter'
```

Go on and try it with an example of trained data from `harry.txt`


## Training

Train with first GPU parse in: `--gpu 0`

+ Basic:

```
$ python train.py --data data/input.txt
```

+ Full options:

```
$ python train.py --data data/harry.txt \
  --result_dir result \
  --resume result/model_epoch_{n} \
  --batch_size 20 \
  --bprop_len 35 \
  --n_units 128 \
  --n_epochs 30 \
  --gpu 0
```

Result will be stored in `result` directory:

+ `model_epoch_{n}`: Serialized model object can be used for resume training or predicted [generating](#generating)

+ `vocab.bin`: Binary object contains vocabulary of training object

## Generating

Generate with first GPU parse in: `--gpu 0`

+ Basic

```
$ python gen.py --model result/model_epoch_{n} \
  --vocab result/vocab.bin \
  --pretext 'Hello'
```

+ Full options:

```
$ python gen.py --model result/model_epoch_{n} \
  --vocab result/vocab.bin \
  --pretext 'Harry' \
  --length 2000 \
  --n_units 128 \
  --gpu 0
```

Result is putted into `stdout`, you can redirect it to a file with a simple trick:

```
$ python gen.py --model result/model_epoch_{n} \
    --vocab result/vocab.bin \
    > sample.txt
```

## References

Code is heavily inspired from these sources:

+ First char-rnn implementation written with Chainer https://github.com/yusuketomoto/chainer-char-rnn

+ Chainer's PTB example https://github.com/pfnet/chainer/tree/master/examples/ptb
