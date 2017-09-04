from torchtext import data
from torchtext import datasets

import util
import streamtologger
import os

import torch
import torch.nn as nn
import torch.optim as O
from torch.autograd import Variable
from torch.nn.utils import rnn

import time
import glob
import sys
from random import randint

# Takes a batch and the lengths of each sequence and returns the PackSequence and the sorted indices
def create_sequence(batch, lengths, batch_first=False):

    batch_size = batch.size(0) if batch_first else batch.size(1)
    sorted_indices = sorted(range(batch_size), key=(lambda k: lengths[k]), reverse=True)
    sorted_lengths = [lengths[i] for i in sorted_indices]
    if batch_first:
        sorted_batch = torch.stack([batch[i] for i in sorted_indices])
    else:
        sorted_batch = torch.stack(
                [batch[:, i]for i in sorted_indices],
                dim=1
        )
    seq = rnn.pack_padded_sequence(sorted_data, sorted_lengths, batch_first=batch_first)
    return seq, sorted_indices

# Given *only* the hiddens of a PackedSequence output RNN, sort them back to original order
def recover_order_hiddens(batch, sorted_indices):

    original_indices = [i for i, _ in sorted(enumerate(sorted_indices), key=(lambda kv: kv[1]))]
    original_batch = torch.stack([batch[i] for i in original_indices])
    return original_batch


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.batch_sizes[0]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(torch.zeros(*state_shape)).cuda()
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1]


class SNLIClassif(nn.Module):

    def __init__(self, config):

        super(SNLIClassif, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = Encoder(config)
        seq_in_size = 4*config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        self.mlp = nn.Sequential(
            nn.Linear(seq_in_size, config.d_hidden_mlp),
            nn.ReLU(),
            nn.Linear(config.d_hidden_mlp, config.d_hidden_mlp),
            nn.ReLU(),
            nn.Linear(config.d_hidden_mlp, config.d_out))

    def forward(self, batch):

        seq_lens_premise = util.seq_lens_from_batch(batch.premise, self.config.pad_vocab_index)
        embeded_premise = self.embed(batch.premise)
        if self.config.fix_emb:
            embeded_premise = Variable(embeded_premise.data)
        packed_premise, sorted_indices_premise = create_sequence(embeded_premise, seq_lens_premise)
        lstm_output_premise = self.encoder(packed_premise)
        premise = recover_order_hiddens(lstm_output_premise, sorted_indices_premise)

        seq_lens_hypothesis = util.seq_lens_from_batch(batch.hypothesis, self.config.pad_vocab_index)
        embeded_hypothesis = self.embed(batch.hypothesis)
        if self.config.fix_emb:
            embeded_hypothesis = Variable(embeded_hypothesis.data)
        packed_hypothesis, sorted_indices_hypothesis = create_sequence(embeded_hypothesis, seq_lens_hypothesis)
        lstm_output_hypothesis = self.encoder(packed_hypothesis)
        hypothesis = recover_order_hiddens(lstm_output_hypothesis, sorted_indices_hypothesis)
        
        scores = self.mlp(torch.cat([premise, hypothesis, premise - hypothesis, premise * hypothesis], 1))
        return scores

args = util.get_args()
util.makedirs(args.save_path)
current_run_dir = args.save_path + "/" + time.strftime("%d:%m") + time.strftime("%H:%M:%S") + args.save_title
util.makedirs(current_run_dir)
streamtologger.redirect(target=current_run_dir + '/log.txt')

torch.cuda.set_device(args.gpu)

inputs = data.Field(lower=args.lower, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(vectors=args.word_vectors)
        util.makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=args.gpu)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
config.pad_vocab_index = inputs.vocab.stoi["<pad>"]

print("Config:", config)

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = SNLIClassif(config)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()

criterion = nn.CrossEntropyLoss()
opt = O.RMSprop(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss Test/Loss     Accuracy  Dev/Accuracy Test/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:8.6f},{:12.4f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
print(header)

for epoch in range(args.epochs):
    start_epoch = time.time()
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        iterations += 1
        # decay lr periodically
        if iterations != 0 and iterations % args.lr_decay_iter == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= args.lr_decay_rate
                print("learning rate decayed to ", param_group['lr'])
                
        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(current_run_dir, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch(); test_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(answer, dev_batch.label)
                if dev_batch_idx == 0:
                    util.print_example(dev_batch, inputs.vocab.itos, answers.vocab.itos, answer, batch.batch_size)
            dev_acc = 100. * n_dev_correct / len(dev)

            # calculate accuracy on test set, only in order to compare with the paper results. Not used for finding best model.
            n_test_correct, test_loss = 0, 0
            for test_batch_idx, test_batch in enumerate(test_iter):
                 answer = model(test_batch)
                 n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
                 test_loss = criterion(answer, test_batch.label)
            test_acc = 100. * n_test_correct / len(test)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], test_loss.data[0], train_acc, dev_acc, test_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(current_run_dir, 'best_snapshot_on_dev')
                snapshot_path = snapshot_prefix + '_devacc_{}_testacc{}__iter_{}_model.pt'.format(dev_acc, test_acc, iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))
    print("time for epoch" + str(epoch) + "  " + util.pretty_duration(time.time() - start_epoch))

