from enum import Enum

class Base_class():
    def __init__(self):
        self.phase = None
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Phase(Enum):
    train = 1
    test = 2
    validation = 3




