import os
import random


def train_test_split():
  names = os.listdir('./data/wav')
  random.seed(9608)
  random.shuffle(names)
  train_ratio = 0.8
  train_names = names[:int(round(len(names) * train_ratio))]
  test_names = names[len(train_names):]
  with open('./data/train.txt', 'w') as f:
    for name in train_names:
      f.write(name + '\n')
  with open('./data/test.txt', 'w') as f:
    for name in test_names:
      f.write(name + '\n')


if __name__ == '__main__':
  train_test_split()
