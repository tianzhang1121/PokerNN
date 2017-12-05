import numpy as np
import random
import uuid

from collections import OrderedDict
from threading import Thread, Lock
from xmlrpc.server import SimpleXMLRPCServer

from .table import Table, TableProxy
from .nn import NeuralNetwork
from .playercontrol import PlayerControl, PlayerControlProxy

class Teacher(Thread):
    def __init__(self, instance_num, seats, n_games, quiet = False, weights = None, biases = None):
        super(Teacher, self).__init__()

        self.instance_num = instance_num
        self.seats = seats
        self.n_games = n_games
        self.host = '0.0.0.0'
        self.port = 8000 + instance_num * 100
        self.table = TableProxy(Table(instance_num, seats, quiet, True))
        self.players = []

        self.add_checkcallbot()
        self.add_nncontroller(weights, biases)
        for i in range(3, seats+1):
            self.add_randombot(i)

    def run(self):
        games_played = 0

        while games_played < self.n_games:
            self.reset_game()
            self.table.run_game()
            games_played += 1
        self.end_game()
        print('finished')

    # cleanup
    '''
    def add_winner(self, winner_uuid):
    '''
    '''
    def child(self, p1, p2):
        child_uuid = uuid.uuid4()
        try:
            weight1 = np.load(NeuralNetwork.SAVE_DIR + p1 + '_weights.npy')
            weight2 = np.load(NeuralNetwork.SAVE_DIR + p2 + '_weights.npy')
            biases1 = np.load(NeuralNetwork.SAVE_DIR + p1 + '_biases.npy')
            biases2 = np.load(NeuralNetwork.SAVE_DIR + p2 + '_biases.npy')

            child_weights = average_arrays(weight1, weight2)
            child_biases = average_arrays(biases1, biases2)

            np.save(NeuralNetwork.SAVE_DIR + str(child_uuid) + '_weights.npy', child_weights)
            np.save(NeuralNetwork.SAVE_DIR + str(child_uuid) + '_biases.npy', child_biases)
        except:
            pass

        return child_uuid
    '''
    def end_game(self):
        for p in self.players:
            p.quit()

    def reset_game(self):
        for p in self.players:
            p.rejoin()

    def add_checkcallbot(self):
        self.players.append(PlayerControlProxy(PlayerControl('localhost', self.port+1, 2, instance_num, True, 2)))

    def add_randombot(self, i):
        # random bot
        self.players.append(PlayerControlProxy(PlayerControl('localhost', self.port+i, i, instance_num, True, 3)))

    def add_nncontroller(self, weights = None, biases = None):
        self.players.append(PlayerControlProxy(PlayerControl('localhost', self.port+2, 0, instance_num, True, 0, weights, biases)))

class TeacherProxy(object):
    def __init__(self, teacher):
        self._quit = False

        self._teacher = teacher
        self.server = SimpleXMLRPCServer((self._teacher.host, self._teacher.port), logRequests=False, allow_none=True)
        self.server.register_instance(self, allow_dotted_names=True)
        Thread(target = self.run).start()

    def run(self):
        while not self._quit:
            self.server.handle_request()

    def quit(self):
        self._quit = true
    '''
    def add_winner(self, winner_uuid):
        self._teacher.add_winner(winner_uuid)
    '''
