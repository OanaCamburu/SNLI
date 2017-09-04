import os
import torch
from argparse import ArgumentParser
from random import randint


def _get_sentence_from_indices(dict, tensor_indices):
    s = ''
    n = len(tensor_indices)
    for i in range(n):
        s = s + ' ' + dict[tensor_indices.data[i]]
    return s    


## Assumes once we have a <pad> the following will be <pad>s
def seq_lens_from_batch(batch, pad_vocab_index, batch_first=False):
    assert(batch.dim() == 2)
    dim = 0
    max_seq_length = batch.size()[0]
    if batch_first:
        dim = 1
        max_seq_length = batch.size()[1]
    seq_lens = torch.max(batch.eq(pad_vocab_index), dim)[1]
    assert(seq_lens.dim() == 1)
    # but the sentences with no <pad> will appear with length 1, change that to max_seq_length
    seq_lens[seq_lens.eq(1)] = max_seq_length
    return list(seq_lens.data.int())

def sort_batch(batch, seq_lens, batch_first=False):
    assert(batch.dim() == 2)
    assert(seq_lens.dim() == 1)
    # sort decresingly the seq_lens
    sorted, indices = seq_lens.sort(dim=0, descending=True)
    return batch[:, indices], sorted, indices


def print_example(batch, input_vocab, answer_vocab, answer, batch_size):
    example_idx = randint(0, batch_size-1)
    print("Example")
    print("premise: ", _get_sentence_from_indices(input_vocab, batch.premise[:,example_idx]))
    print("hypothesis: ", _get_sentence_from_indices(input_vocab, batch.hypothesis[:,example_idx]))
    print("gold label: ", _get_sentence_from_indices(answer_vocab, batch.label[example_idx]))
    print("predicted label: ", answer_vocab[torch.max(answer, 1)[1].view(batch.label.size()).data[example_idx]])


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

def pretty_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--d_hidden_mlp', type=int, default=1024)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--weight_decay', type=float, default=.000006)
    parser.add_argument('--lr_decay', action='store_false', dest='lr_decay')
    parser.add_argument('--lr_decay_rate', type=float, default=.75)
    parser.add_argument('--lr_decay_iter', type=int, default=10000)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.06)
    parser.add_argument('--bidirectional', action='store_true', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results_padding')
    parser.add_argument('--save_title', type=str, default='')
    #parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
