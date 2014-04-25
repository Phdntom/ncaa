
import numpy as np
import os
import glob

from ncaa_lib.split_seasons import split_seasons
from ncaa_lib.season_accumulator import prep_rating_matrix
from ncaa_lib.matrix_factorization import grad_descent_factorization


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

def factorize_seasons(parent_folder, type_name):

    matrix_name = "Ymatrix.npy"

    dir_names = []
    with change_dir(parent_folder):
        dir_names = glob.glob( type_name + "*" )
        dir_names.sort()
        print("Processing matrices")
        for dirname in dir_names:
            path = dirname + '/' + matrix_name
            print(path)
            Y = np.load(path)
            print( Y.shape)
            print(Y)
            print(sum(sum(Y != 0)))
            nP, nQ = grad_descent_factorization(Y, K=10, steps=5000,
                                                alpha=0.01, lamda=0.002,
                                                disp=True)
            eY = np.dot(nP, nQ.T)
            print(max(abs(eY * (Y != 0).astype(int) - Y).flatten()) )
            with change_dir(dirname):
                np.save("eYmatrix", eY)

    
def main():

    parent_folder = "metadata"
    type_name = "season"

    season_fname = "../data/regular_season_results.csv"
    split_seasons(season_fname, parent_folder, type_name)

    team_fname = "../data/teams.csv"
    prep_rating_matrix(team_fname, parent_folder, type_name)

    factorize_seasons(parent_folder, type_name)
    






if __name__ == "__main__":

    main()
