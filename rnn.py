import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.l2_reg = 1e-3
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        rnn_out, _ = self.rnn(inputs)
        out = self.W(rnn_out[-1])
        predicted_vector = self.softmax(out)
        return predicted_vector


def load_data(train_data, val_data, test_data=None):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    if test_data:
        with open(test_data) as test_f:
            test = json.load(test_f)
    else:
        test = []

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    tes = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in test] if test else []

    return tra, val, tes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default=None, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print(f"Training started for epoch {epoch + 1}")
        correct = 0
        total = 0
        minibatch_size = 32
        N = len(train_data)

        train_loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            train_loss_total += loss.item()
            loss_count += 1
            loss.backward()
            optimizer.step()

        train_loss_avg = train_loss_total / loss_count
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Training loss for epoch {epoch + 1}: {train_loss_avg}")
        training_accuracy = correct / total

        model.eval()
        correct = 0
        total = 0
        valid_loss_total = 0

        print(f"Validation started for epoch {epoch + 1}")
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)

            example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
            predicted_label = torch.argmax(output)

            correct += int(predicted_label == gold_label)
            total += 1
            valid_loss_total += example_loss.item()

        validation_accuracy = correct / total
        validation_loss_avg = valid_loss_total / total
        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {validation_accuracy}")
        print(f"Validation loss for epoch {epoch + 1}: {validation_loss_avg}")

        if validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training stopped to avoid overfitting.")
            print(f"Best validation accuracy: {last_validation_accuracy}")
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = training_accuracy

        epoch += 1

    # Test evaluation (if test data is provided)
    if args.test_data:
        model.eval()
        correct = 0
        total = 0

        print("========== Evaluating on Test Data ==========")
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding.get(i.lower(), word_embedding['unk']) for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)

            correct += int(predicted_label == gold_label)
            total += 1

        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy}")
