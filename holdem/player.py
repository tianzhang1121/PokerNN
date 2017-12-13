from .holdemai import HoldemAI
from deuces.deuces import Card
import uuid
import numpy as np
class Player(object):
    def __init__(self, playerID, emptyplayer = False, ai_flag = False, ai_type = -1, weights = None, biases = None, stack = 2000):
        self.playerID = playerID

        self.hand = []
        self.stack = stack
        self.currentbet = 0
        self.lastsidepot = 0
        self._seat = -1
        self.handrank = None

        # flags for table management
        self.emptyplayer = emptyplayer
        self.betting = False
        self.isallin = False
        self.playing_hand = False
        self.playedthisround = False
        self.sitting_out = True

        self._ai_flag = ai_flag

        if self._ai_flag:
            self._ai_type = ai_type
            if self._ai_type == 0:
                self.ai = HoldemAI(uuid.uuid4(), weights, biases)

    def get_seat(self):
        return self._seat

    def set_seat(self, value):
        self._seat = value

    def reset_hand(self):
        self.hand=[]
        self.playedthisround = False
        self.betting = False
        self.isallin = False
        self.currentbet = 0
        self.lastsidepot = 0
        self.playing_hand = (self.stack != 0)

    def bet(self, bet_size):
        self.playedthisround = True
        if not bet_size:
            return
        self.stack -= (bet_size - self.currentbet)
        self.currentbet = bet_size
        if self.stack == 0:
            self.isallin = True

    def refund(self, ammount):
        self.stack += ammount

    def player_state(self):
        return (self.get_seat(), self.stack, self.playing_hand, self.betting, self.playerID)

    def get_ai_type(self):
        return self._ai_type

    def get_ai_id(self):
        if self._ai_type == 0:
            return str(self.ai.networkID)
        else:
            return self._ai_type

    def save_ai_state(self):
        if self._ai_flag and self._ai_type == 0:
            print('AI type NEURAL NETWORK won (', self.get_ai_id(), ')')
            # self.writer.write([self.ai.networkID, consec_wins])
            self.ai.save()
        else:
            print('AI type ', self._ai_type, 'won')

    def delete_ai(self):
        if self._ai_type == 0:
            self.ai.delete()

    def new_ai(self, ai_id):
        if ai_id == 'unchanged':
            pass
        else:
            self.ai = HoldemAI(ai_id) # defaults to random network if ai_id not recognized

    def new_ai_type(self, ai_type):
        self._ai_type = ai_type

    def reset_stack(self):
        self.stack = 2000

    def print_table(self, table_state):
        print('Stacks:')
        players = table_state.get('players', None)
        for player in players:
            print(player[4], ': ', player[1], end='')
            if player[2] == True:
                print('(P)', end='')
            if player[3] == True:
                print('(Bet)', end='')
            if player[0] == table_state.get('button'):
                print('(Button)', end='')
            if players.index(player) == table_state.get('my_seat'):
                print('(me)', end='')
            print('')

        print('Community cards: ', end='')
        Card.print_pretty_cards(table_state.get('community', None))
        print('Pot size: ', table_state.get('pot', None))

        print('Pocket cards: ', end='')
        Card.print_pretty_cards(table_state.get('pocket_cards', None))
        print('To call: ', table_state.get('tocall', None))

    def update_localstate(self, table_state):
        self.stack = table_state.get('stack')
        self.hand = table_state.get('pocket_cards')

    # cleanup
    def player_move(self, table_state):
        self.update_localstate(table_state)
        bigblind = table_state.get('bigblind')
        tocall = min(table_state.get('tocall', None), self.stack)
        minraise = table_state.get('minraise', None)
        # print('minraise ', minraise)
        # move_tuple = ('Exception!',-1)

        # ask this human meatbag what their move is
        if not self._ai_flag:
            self.print_table(table_state)
            if tocall == 0:
                print('1) Raise')
                print('2) Check')
                try:
                    choice = int(input('Choose your option: '))
                except:
                    choice = 0
                if choice == 1:
                    choice2 = int(input('How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                    while choice2 < minraise:
                        choice2 = int(input('(Invalid input) How much would you like to raise? (min = {}, max = {})'.format(minraise,self._stack)))
                    move_tuple = ('raise',choice2)
                elif choice == 2:
                  move_tuple = ('check', 0)
                else:
                    move_tuple = ('check', 0)
            else:
                print('1) Raise')
                print('2) Call')
                print('3) Fold')
                try:
                    choice = int(input('Choose your option: '))
                except:
                    choice = 0
                if choice == 1:
                    choice2 = int(input('How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                    while choice2 < minraise:
                        choice2 = int(input('(Invalid input) How much would you like to raise to? (min = {}, max = {})'.format(minraise,self._stack)))
                    move_tuple = ('raise',choice2)
                elif choice == 2:
                    move_tuple = ('call', tocall)
                elif choice == 3:
                   move_tuple = ('fold', -1)
                else:
                    move_tuple = ('call', tocall)

        # feed table state to ai and get a response
        else:
            # neural network output
            if self._ai_type == 0:
                # neural network output
                move_tuple = self.ai.act(table_state)

            elif self._ai_type == 1:
                # check/fold bot
                if tocall > 0:
                    move_tuple = ('fold',-1)
                else:
                    move_tuple = ('check', 0)
            elif self._ai_type == 2:
                # check/call bot
                if tocall > 0:
                    move_tuple = ('call',tocall)
                else:
                    move_tuple = ('check', 0)
            else:
                if tocall >0:
                    # 0 - Raise
                    # 1 - Call
                    # 2 - Fold
                    move_idx = np.random.randint(0,2)
                    if move_idx == 0:
                        try:
                            bet_size = np.random.randint(minraise, self.stack)
                            bet_size -= bet_size % bigblind
                        except:
                            bet_size = self.stack
                        if bet_size <= tocall:
                            move_tuple = ('call', tocall)
                        else:
                            move_tuple = ('raise', bet_size)
                    elif move_idx == 1:
                        move_tuple = ('call', tocall)
                    else:
                        move_tuple = ('fold', -1)
                else:
                    # 0 - Raise
                    # 1 - Check
                    move_idx = np.random.randint(0,1)
                    if move_idx == 0:
                        try:
                            bet_size = np.random.randint(minraise, self.stack)
                            bet_size -= bet_size % bigblind
                        except:
                            bet_size = self.stack
                        move_tuple = ('raise', bet_size)
                    else:
                        move_tuple = ('check',0)

        return move_tuple

