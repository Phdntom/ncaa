from __future__ import division, print_function
import numpy as np

import glob as glob
from collections import defaultdict
import json as json

from ncaa_lib.os_manip_utilities import change_dir
from ncaa_lib.scoring import game_score, point_diff
from ncaa_lib.split_seasons import split_seasons
from ncaa_lib.TeamStats import team_stats

def record_game(teams, win_id, win_score, los_id, los_score, home=None):
    '''
    # home is None:     home/away ignored, stats grouped together in class def
    # home is True:     the win_id is the home team
    # home is False:    the win_id is the away team
    '''

    if win_id not in teams:
        teams[win_id] = team_stats(win_id)
    if los_id not in teams:
        teams[los_id] = team_stats(los_id)

    if home is None:    # ignore home/away distinction
        teams[win_id].add_game(win_score, los_id, los_score)
        teams[los_id].add_game(los_score, win_id, win_score)
    elif home:          # home won
        teams[win_id].add_home_game(win_score, los_id, los_score)
        teams[los_id].add_away_game(los_score, win_id, win_score)
    else:               # away won
        teams[win_id].add_away_game(win_score, los_id, los_score)
        teams[los_id].add_home_game(los_score, win_id, win_score)   

def play_games(fobj, home_away=False):
    '''

    Parameters
    ----------
    fobj:       valid file object contain lines with game data
    home_away:  determines if home/away field is used to parse game data

    Returns
    -------
    teams:      dictionary of team_stats objects
    
    '''
    # Dict of team_stats objects
    teams = {}

    titles = fobj.readline()
    for game in fobj:
        fields = game.split(',')
        season, day = fields[0], int(fields[1])
        win_id, win_score = int(fields[2]), int(fields[3])
        los_id, los_score = int(fields[4]), int(fields[5])
        if home_away:
            home_won = fields[6] == 'H'
            record_game(teams, win_id, win_score, los_id, los_score, home_won)
        else:
            record_game(teams, win_id, win_score, los_id, los_score)

    return teams

def chess_rank(teams, steps=10):

    # Set all teams current rating advantage
    for team_id in teams:
        teams[team_id].set_rating_adv()

    ratings_dict = {}
    for i in range(steps):
        # Fix ratings for each team at the currnet iteration
        for team_id in teams:
            ratings_dict[team_id] = teams[team_id].rating

        # Find the opponent rating and set the rating for each team
        for team_id in teams:
            opp_rating = teams[team_id].avg_opponent_rating(ratings_dict)
            teams[team_id].opponent_rating = opp_rating
            rating = opp_rating + teams[team_id].rating_adv
            teams[team_id].rating = rating

        # Calibrate back to a rating mean of 50
        mean_rating = np.mean([teams[team_id].rating for team_id in teams])
        adjust = 50 - mean_rating
        for team_id in teams:
            teams[team_id].rating += adjust

    final_ratings = {}
    for team_id in teams:
         final_ratings[team_id] = teams[team_id].rating

    return final_ratings

def home_away_chess_rank(teams, steps=50):

    # Set all teams current rating advantage
    for team_id in teams:
        teams[team_id].set_home_away_rating_adv()

    #teams_list = teams.values()

    home_ratings = {}
    away_ratings = {}
    for i in range(steps):
        # Fix ratings for each team at the currnet iteration
        for team_id in teams:
            home_ratings[team_id] = teams[team_id].home_rating
            away_ratings[team_id] = teams[team_id].away_rating

        # Find the opponent rating and set the rating for each team
        for team_id in teams:
            home_opp_rating = teams[team_id] \
                        .avg_home_opp_rating(away_ratings)
            teams[team_id].home_opp_rating = home_opp_rating
            home_rating = home_opp_rating + teams[team_id].home_rating_adv
            teams[team_id].home_rating = home_rating

            away_opp_rating = teams[team_id] \
                        .avg_away_opp_rating(home_ratings)
            teams[team_id].away_opp_rating = away_opp_rating
            away_rating = away_opp_rating + teams[team_id].away_rating_adv
            teams[team_id].away_rating = away_rating

        # Calibrate
        mean_home_rating = np.mean(
                    [teams[team_id].home_rating for team_id in teams])
        mean_away_rating = np.mean(
                    [teams[team_id].away_rating for team_id in teams])
        total_adjust = 50 - (mean_home_rating + mean_away_rating) / 2
        for team_id in teams:
            teams[team_id].home_rating += total_adjust
            teams[team_id].away_rating += total_adjust
        print(np.mean([teams[team_id].home_rating for team_id in teams]))
        print(np.mean([teams[team_id].away_rating for team_id in teams]))
        print()

    ratings_home = {}
    ratings_away = {}
    for team_id in teams:
         ratings_home[team_id] = teams[team_id].home_rating
         ratings_away[team_id] = teams[team_id].away_rating

    return ratings_home, ratings_away
    
def log_loss(y, y_pred):

    ln = np.log
    
    return  -np.mean(y * ln(y_pred) + (1 - y) * ln(1 - y_pred))

def score_predictions(results, all_ratings):

    y_pred = np.zeros(63)

    logLoss = []

    with open(results) as fobj:
        titles = fobj.readline()
        index = 0
        for line in fobj:
            row = line.split(',')
            season, day = row[0], row[1]

            win_id = int(row[2])
            los_id = int(row[4])
            rating_win = all_ratings[season][win_id]
            rating_los = all_ratings[season][los_id]

            y_pred[index] = game_score( (rating_win - rating_los) )
            index += 1
            if index > 62:
                index = 0
                logLoss.append( log_loss(1, y_pred))
    print(np.mean(logLoss))

def score_home_away(param, results_file, home_ratings, away_ratings):

    y_pred = np.zeros(63)

    logLoss = []

    with open(results_file) as fobj:
        titles = fobj.readline()
        index = 0
        for line in fobj:
            row = line.split(',')
            season, day = row[0], row[1]

            win_id = int(row[2])
            los_id = int(row[4])
            rating_win_home = home_ratings[season][win_id]
            rating_los_home = home_ratings[season][los_id]

            rating_win_away = away_ratings[season][win_id]
            rating_los_away = away_ratings[season][los_id]

            y_home = game_score(rating_win_home - rating_los_home)
            y_away = game_score(rating_win_away - rating_los_away)

            
            rating_diff = param * (rating_win_home - rating_los_home) \
                        + (1 - param) * (rating_win_away - rating_los_away)

            
            # weighted geometric
            score = np.exp( param * np.log(y_home) + (1-param) * np.log(y_away))

            # 
            y_pred[index] = score
            index += 1
            if index > 62:
                index = 0
                logLoss.append( log_loss(1, y_pred))
    print(param, np.mean(logLoss))


if __name__ == "__main__":
    np.seterr('raise')
    home_away = True

    parent_folder = "metadata"
    base_dir = "season"

    season_file = "../data/regular_season_results.csv"
    split_seasons(season_file, parent_folder, base_dir)

    season_dirs = glob.glob("./{0}/{1}?".format(parent_folder,base_dir) )
    season_dirs.sort()
    all_season_ratings = {}
    all_home_ratings = {}
    all_away_ratings = {}

    for season in season_dirs:
        fname = "{0}{1}.csv".format(base_dir,season[-1])

        with change_dir(season), open(fname, "r") as fobj:
            teams = play_games(fobj, home_away)
            if home_away:
                home_ratings, away_ratings = home_away_chess_rank(teams)
                all_home_ratings[season[-1]] = home_ratings
                all_away_ratings[season[-1]] = away_ratings
            
            else:
                final_ratings = chess_rank(teams)
                all_season_ratings[season[-1]] = final_ratings
           
            #rating_list = [teams[key].rating for key in teams]
            #rating_list.sort()

    results_file = "../data/tourney_results.csv"
    if home_away:
        print("SINGLE RATING",0.5467468)
        for param in np.arange(.1,.91,0.05):
            score_home_away(param, results_file, \
                            all_home_ratings, all_away_ratings)
        
    else:
        print(len(all_season_ratings))
        score_predictions(results_file, all_season_ratings)






