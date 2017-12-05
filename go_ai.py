from holdem import Table, TableProxy, PlayerControl, PlayerControlProxy, Teacher, TeacherProxy
import argparse
import time

seats = 8

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_instances', type=int, default=10)
    parser.add_argument('n_games', type=int, default=1)
    #termination criteria
    parser.add_argument('n_gens', type=int, default = 1000)
    parser.add_argument('min_fitness', type=float, default = 0.0)

    parser.add_argument('--quiet', dest='quiet', action='store_true')
    args = parser.parse_args()

    #create multiple teachers and proxys
    teachers = []
    teacher_proxies = []
    for i in range(args.n_instances):
        teachers.append(Teacher(i, seats, args.n_games, args.quiet))
        teacher_proxies.append(TeacherProxy(teachers[i]))
        teachers[i].start()

    for i in range(len(teachers)):
        teachers[i].join()