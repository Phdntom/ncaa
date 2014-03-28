from __future__ import division, print_function
import numpy as np
import os
import glob as glob
from collections import defaultdict
import json as json
from matrix_factorization import grad_descent_factorization            

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
    
def game_score(rating_diff):

    return 1 / ( 1 + 10 ** (-rating_diff / 15) )

def game_rating(score_diff):

    return -15 * np.log10( 1 / score_diff - 1)

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

def process_seasons(parent_folder, sub_dir):

    fname = "data/teams.csv"
    idnameMap = build_teams(fname)
    print(idnameMap)

    all_ratings = {}

    with change_dir(parent_folder):
        dir_names = glob.glob(sub_dir + '*')
        dir_names.sort()
        print("Processing directories...")
        for dirname in dir_names:
            season = dirname[-1]
            print(season)
            fname = dirname + '.csv'
            with change_dir(dirname), open(fname,"r") as fobj:
                all_ratings[season] = converge_ratings(fobj, idnameMap)

    return all_ratings

def score_opponents(fobj):

    scores = defaultdict(list)
    opponents = defaultdict(list)

    titles = fobj.readline().split(',')
    game_count = 0
    for l, game in enumerate(fobj):
        fields = game.split(',')
        season, day = fields[0], int(fields[1])
        win_id, win_score = int(fields[2]), int(fields[3])
        los_id, los_score = int(fields[4]), int(fields[5])
        home_won = fields[6] == 'H'
        opponents[win_id].append(los_id)
        opponents[los_id].append(win_id)
        scores[win_id].append(game_score(win_score - los_score))
        scores[los_id].append(game_score(los_score - win_score))
        game_count += 1
    print("game count", game_count)
    return scores, opponents

def intialize_ratings(scores, opponents):
    '''
    '''
    ratings = {}
    rating_count = 0
    for team in opponents:
        Y = defaultdict(float)
        N = defaultdict(int)
        for opp, score in zip(opponents[team],scores[team]):
            
            Y[opp] = (Y[opp] * N[opp] + score) / (N[opp] + 1)
            N[opp] += 1
            rating_count += 1
        R = {}
        for opp in Y:
            R[opp] = 50 + game_rating(Y[opp])
        ratings[team] = R

    return ratings

def build_matrix(ratings, idToIndex, indexToID):

    N = len(ratings)
    Y = np.zeros(shape=(N,N))

    for team in ratings:
        team_idx = idToIndex[team]
        for op in ratings[team]:
            op_idx = idToIndex[op]
            Y[team_idx][op_idx] = ratings[team][op]

    np.save("RatingMatrix",Y)

    return Y

def converge_ratings(fobj, idnameMap):
    '''
    '''
    np.set_printoptions(precision=3,threshold=9,edgeitems=4,linewidth=100, suppress=True)

    scores, opponents = score_opponents(fobj)
    ratings = intialize_ratings(scores, opponents)

    idToIndex = defaultdict(int)
    indexToID = defaultdict(str)
    for index, idnum in enumerate(ratings):
        idToIndex[idnum] = index
        indexToID[index] = idnum

    R_matrix = build_matrix(ratings, idToIndex, indexToID)
    print(R_matrix)
    nP, nQ = grad_descent_factorization(R_matrix, K=15, steps=5000,
                                        alpha=0.001, lamda=0.005)
    eR = np.dot(nP, nQ.T)
    print(np.mean(abs(eR * (R_matrix != 0).astype(int) - R_matrix).flatten()) )
    print(eR)
    np.save("eRatingMatrix", eR)

    final_R = eR.copy()
    final_R[R_matrix != 0] = 0
    final_R += R_matrix
    print("FinalR\n",final_R)

    predictions = defaultdict(lambda: defaultdict(float))
    S_matrix = np.zeros(shape=final_R.shape)
    S_matrix2 = np.zeros(shape=final_R.shape)
    for i in range(len(final_R)):
        for j in range(i+1):
            if i != j:
                valij =     game_score(final_R[i][j] - 50)
                valji = 1 - game_score(final_R[j][i] - 50)
                val = (valij + valji) / 2
                S_matrix[i][j] = val
                S_matrix[j][i] = 1 - val
            else:
                S_matrix[i][j] = game_score(final_R[i][j] - 50)
    print(S_matrix)
    for i, row in enumerate(S_matrix):
        id1 = indexToID[i]
        for j, score in enumerate(row):
            id2 = indexToID[j]
            predictions[id1][id2] = score

    return predictions
    
def find_probability(season_predictions):

    matrix_name = "RatingMatrix.npy"
    path_sample = "data/sample_submission.csv"
    path_predict = "data/my_predictions.csv"
    game_pred = []
    with open(path_sample,"r") as fin:
        fin.readline()
        for line in fin:
            info, num = line.split(',')
            season, team1, team2 = info.split('_')
            score = season_predictions[season][int(team1)][int(team2)]
            game_pred.append(info + ',' + str(score) )
        print(len(game_pred))
    with open(path_predict,"wb") as fout:
        fout.write("\n".join(game_pred))

if __name__ == "__main__":

    parent_folder = "metadata"
    sub_dir = "season"
    



    season_predictions = process_seasons(parent_folder, sub_dir)

    #with open("scores_dict.json", "wb") as json_obj:
    #    json.dump(season_predictions,json_obj, sort_keys=True, indent=4, separators=(',',': '))

    find_probability(season_predictions)

    #factorize_seasons(parent_folder, sub_dir)










