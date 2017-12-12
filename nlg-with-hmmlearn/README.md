# NLG with hmmlearn

Experimental natural language generation with hidden markov models, using the
excellent [hmmlearn] Python library.

[hmmlearn]: http://hmmlearn.readthedocs.io/en/latest/

## Installing dependencies

Running the programs requires Python (tested on 3.5.2). Install required
modules with:

    $ pip install -r requirements.txt

## Training a model

Example for training a model with 8 hidden states based on given text.

The input text is assumed to have one sentence per line (or segment of
comparable size). We recommend cleaning up the text as much as possible,
leaving only the essential punctuation (if any).

    $ mkdir demo
    $ ./train.py -n 8 -o demo/hmm < ../input/input.txt

## Generating new text

Generate `20` lines of text, `12` words per line, by simulating the hidden
markov model obtained in the previous step:

    $ ./gen.py -l 20 -w 12 demo/hmm.builtin.8

For comparison, try generating new text based solely on the frequency
distribution of words in the input text. `.le` and `.freqdist` files are
generated as part of the training step. (Despite the file names, the number of
hidden states chosen in the training step does not affect their contents.)

    $ ./freq.py -l 20 -w 12 demo/hmm.builtin.8.freqdist demo/hmm.builtin.8.le

Lastly, compare the output with text generated by choosing words from the
input text at random:

    $ ./rnd.py -l 20 -w 12 demo/hmm.builtin.8.le

## License

This project, like hmmlearn, uses the BSD license.