## Char-rnn generation

Character-level language model implementation of Karpathy's char-rnn (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

+ Current version is written with [Pytorch](http://pytorch.org/) and GRU shell

+ `chainer/` contains old version writen with [Chainer](https://chainer.org/) and LSTM shell


### Requirements

+ Python >= 3.4

+ Pytorch


### Train

```sh
python train.py --epoch 5000 --frequency 100 --gpu 0 --source harry.txt
```

Model is saved in `save/char-rnn-gru.pt`


### Generate

```sh
python generate.py --model save/char-rnn-gru.pt --prime "Harry Potter" --len 2000
```
