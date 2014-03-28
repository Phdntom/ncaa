from __future__ import division, print_function
import numpy as np
import os
import glob as glob
from collections import defaultdict
import json as json

def game_score(point_difference):

    return 1 / ( 1 + 10 ** (-point_difference / 15) )

def game_rating(game_score):

    return -15 * np.log10( 1 / game_score - 1)

def build_teams(fname):
    '''
    '''
    id_names = {}
    with open(fname,"r") as fobj:
        fields = (fobj.readline()).split(",")
        for team in fobj:
            team_id, team_name = team.split(",")
            id_names[int(team_id)] = team_name.strip()

    return id_names

class change_dir():
    '''
    '''
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def accumulate_stats(fobj, id_name_map):
    '''
    '''
    # dict of score for a teams list of opponents
    list_gamescores = defaultdict(list)
    # dict of ids for a teams list of opponents
    list_opponents = defaultdict(list)

    titles = fobj.readline().split(',')
    for l, game in enumerate(fobj):
        fields = game.split(',')
        season, day = fields[0], int(fields[1])
        win_id, win_score = int(fields[2]), int(fields[3])
        los_id, los_score = int(fields[4]), int(fields[5])
        home_won = fields[6] == 'H'
        list_gamescores[win_id].append(game_score(win_score - los_score))
        list_gamescores[los_id].append(game_score(los_score - win_score))
        list_opponents[win_id].append(los_id)
        list_opponents[los_id].append(win_id)
        # teams can play each other more than once
        # for now, just taking the mean of the gamescores for Y entry below

    idToIndex = defaultdict(int)
    indexToID = defaultdict(str)
    for index, idnum in enumerate(list_gamescores):
        idToIndex[idnum] = index
        indexToID[index] = idnum

    N_teams = len(list_opponents)
    Y = np.zeros(shape=(N_teams,N_teams))
    R = np.zeros(shape=(N_teams,N_teams))
    for team_id in list_opponents:
        team_idx = idToIndex[team_id]
        opponent_idxs = [idToIndex[op_id] for op_id in list_opponents[team_id]]
        scores = list_gamescores[team_id]
        for opidx,score in zip(opponent_idxs,scores):
            old_Y = Y[team_idx][opidx]
            old_R = R[team_idx][opidx]
            # handles all cases of repeated matchups
            # Y = the mean gamescore of all contests between two teams
            Y[team_idx][opidx] = (old_Y * old_R + score) / (old_R + 1)
            R[team_idx][opidx] += 1

    return Y, idToIndex, indexToID

def sparse_visual(M):

    for i, row in enumerate(M):
        print("i=",i,end=": ")
        for j, col in enumerate(M[i]):
            if M[i][j] != 0:
                print("{0}:{1} ".format(j,M[i][j]), end="")
        print()

def prep_rating_matrix(teamnames, parent_folder, type_name):
    '''
    '''
    id_name_map = build_teams(teamnames)
    print(id_name_map)

    fname = "Ymatrix"
    idname = "idMap.json"
    indexname = "indexMap.json"
    
    dir_names = []
    with change_dir(parent_folder):
        dir_names = glob.glob( type_name + "*" )
        dir_names.sort()
        print("Processing directories...")
        for dirname in dir_names:
            path = dirname + '/' + dirname + '.csv'
            print(dirname, path)
            with open(path, "r") as fobj, change_dir(dirname):
                Y, idMap, indexMap = accumulate_stats(fobj, id_name_map)
                np.save(fname, Y)
                with open(idname, "wb") as fobj:
                    json.dump(idMap, fobj)
                with open(indexname, "wb") as fobj:
                    json.dump(indexMap, fobj)

if __name__ == "__main__":

    fname = "data/teams.csv"
    parent_folder = "metadata"
    type_name = "season"
    prep_rating_matrix(fname, parent_folder, type_name)





    
