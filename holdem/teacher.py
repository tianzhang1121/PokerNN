import numpy as np
import random
import uuid

from collections import OrderedDict
from multiprocessing import Process, Queue

from .table import Table
from .nn import NeuralNetwork

class Teacher(Process):
    def __init__(self, queue, instance_num, seats, n_games, quiet = False, weights = None, biases = None):
        super(Teacher, self).__init__()
        self.queue = queue
        self.instance_num = instance_num
        self.seats = seats
        self.n_games = n_games
        self.table = Table(instance_num, seats, quiet, True)
        #self.players = []
        #self.wins = 0
        self.add_checkcallbot()
        self.add_nncontroller(weights, biases)
        for i in range(3, seats+1):
            self.add_randombot(i)

    def run(self):
        #print(self.instance_num)
        games_played = 0
        val = 0
        wins = 0
        while games_played < self.n_games:
            self.table.reset_stacks()
            place = self.table.run_game()
            if place == 1:
                wins += 1
            val += (1.0 / float(place)) 
            games_played += 1
        fitness = 1.0 - (val / self.n_games)
        self.queue.put([fitness, self.get_weights(), self.get_biases(), wins])
        print("finished playing an instance")
        #print(fitness)
        #print("instance_num" + str(self.instance_num) + ":" + str(wins))

    def add_checkcallbot(self):
        self.table.add_player(2, False, True, 2)

    def add_randombot(self, i):
        # random bot
        self.table.add_player(i, False, True, 3)

    def add_nncontroller(self, weights = None, biases = None):
        self.table.add_player(0, False, True, 0, weights, biases)

    def get_weights(self):
        return self.table.get_ai().get_weights()

    def get_biases(self):
        return self.table.get_ai().get_biases()
