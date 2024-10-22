import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h, dropout_rate=0.1):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        x1 = self.activation(self.W1(input_vector))
        x1 = self.dropout(x1)
        y = self.W2(x1)
        predicted_vector = self.softmax(y.view(-1, self.output_dim))
        return predicted_vector


def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    print(f"Number of records in training data: {len(tra)}")
    print(f"Number of records in validation data: {len(val)}")    

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    print("========== Training for {} epochs ==========".format(args.epochs))
    with open("results.txt", "a") as f:
        f.write("Epoch    Training_accuracy    Training_loss    Training_time    Validation_accuracy    Validation_loss    Validation_time\n")
        start_time_overall = time.time()
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            epoch_loss = 0
            start_time = time.time()
            print("Training started for epoch {}".format(epoch + 1))
            random.shuffle(train_data) 
            minibatch_size = 32
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    epoch_loss += example_loss.item()  # Accumulate loss for this epoch
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()

            epoch_loss /= (N // minibatch_size)  # Average training loss
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Training loss for epoch {}: {:.4f}".format(epoch + 1, epoch_loss))
            print("Training time for this epoch: {}".format(time.time() - start_time))

            # Log training loss
            f.write("{:<9}{:<20.4f}{:<20.4f}{:<17.4f}".format(epoch + 1, correct / total, epoch_loss, time.time() - start_time))

            # Validation
            loss = None
            correct = 0
            total = 0
            valid_epoch_loss = 0
            start_time = time.time()
            print("Validation started for epoch {}".format(epoch + 1))
            minibatch_size = 16 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    valid_epoch_loss += example_loss.item()  # Accumulate validation loss
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size

            valid_epoch_loss /= (N // minibatch_size)  # Average validation loss
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Validation loss for epoch {}: {:.4f}".format(epoch + 1, valid_epoch_loss))
            print("Validation time for this epoch: {}".format(time.time() - start_time))
            f.write("{:<23.4f}{:<20.4f}{:<15.4f}\n".format(correct / total, valid_epoch_loss, time.time() - start_time))

        if args.test_data:
            print("========== Evaluating on Test Data ==========")
            with open(args.test_data) as test_f:
                test_data = json.load(test_f)

            tes = []
            for elt in test_data:
                tes.append((elt["text"].split(), int(elt["stars"] - 1)))
            print(f"Number of records in test data: {len(tes)}")
            
            test_data = convert_to_vector_representation(tes, word2index)

            model.eval()
            correct = 0
            total = 0
            minibatch_size = 16
            N = len(test_data)
            
            with torch.no_grad():
                for minibatch_index in tqdm(range(N // minibatch_size)):
                    loss = None
                    for example_index in range(minibatch_size):
                        input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector)
                        correct += int(predicted_label == gold_label)
                        total += 1

            test_accuracy = correct / total
            print(f"Test accuracy: {test_accuracy:.4f}")
