from __future__ import division, print_function
import numpy as np

from scoring import game_score, point_diff

class team_stats():
    '''
    '''
    def __init__(self, idnum):
        # the teams id in the data
        self.idnum = idnum

        # Initial rating (I don't think this matters)
        self.rating = 50 #
        self.home_rating = 50
        self.away_rating = 50

        # ratings and rating advantages
        self.rating_adv = None
        self.home_rating_adv = None
        self.away_rating_adv = None

        self.opponent_rating = None
        self.home_opp_rating = None
        self.away_opp_rating = None

        # track score and opponent ids for each games
        self.game_scores = []
        self.home_scores = []
        self.away_scores = []

        self.opponent_ids = []
        self.home_opps = []
        self.away_opps = []

    def __str__(self):
        if len(self.opponent_ids) == 0:
            return self._home_away_str()

        results = zip(self.opponent_ids, self.game_scores)
        #print(results)
        lines = ["{0}: {1}".format(x[0],x[1]) for x in results]
        team_head = "team_id= {0}, rating= {1}, " \
                    .format(self.idnum, self.rating) \
                    + "opp_rating= {0}, rating_adv= {1}:\n\t" \
                      .format(self.opponent_rating,self.rating_adv)

        return team_head # + "\n\t".join(lines)

    def _home_away_str(self):
        home_results = zip(self.home_opps, self.home_scores)
        away_results = zip(self.away_opps, self.away_scores)
        #print(results)
        home_line = ["{0}: {1}".format(x[0],x[1]) for x in home_results]
        away_line = ["{0}: {1}".format(x[0],x[1]) for x in away_results]
        team_head = "team_id= {0}, HR= {1}, AR = {2}\n " \
                    .format(self.idnum, self.home_rating, self.away_rating) \
                    + "h_opp_rating= {0}, home_rating_adv= {1}:\n" \
                      .format(self.home_opp_rating,self.home_rating_adv) \
                    + "a_opp_rating= {0}, away_rating_adv= {1}:\n" \
                      .format(self.away_opp_rating,self.away_rating_adv)

        return team_head  + "home games:\n\t" + "\n\t".join(home_line) \
                          + "\naway_games:\n\t" + "\n\t".join(away_line)

    def set_rating_adv(self):
        avg_gs = np.mean(self.game_scores)
        self.rating_adv = point_diff(avg_gs)

    def avg_opponent_rating(self, rating_dict):
        return np.mean([rating_dict[op_id] for op_id in self.opponent_ids])

    def add_game(self, team_score, opp_id, opp_score):
        self.opponent_ids.append(opp_id)
        self.game_scores.append(game_score(team_score - opp_score))

    # home away methods
    def set_home_away_rating_adv(self):
        if not self.home_scores:
            self.home_rating_adv = 0
        else:
            avg_gs = np.mean(self.home_scores)
            self.home_rating_adv = point_diff(avg_gs)

        if not self.away_scores:
            self.away_rating_adv = 0
        else:
            avg_gs = np.mean(self.away_scores)
            self.away_rating_adv = point_diff(avg_gs)

    def avg_home_opp_rating(self, away_dict):
        if not self.home_opps:
            return 50
    # HERE HERE HERE
        return np.mean([away_dict[op_id] for op_id in self.home_opps])

    def avg_away_opp_rating(self, home_dict):
        if not self.away_opps:
            return 50

        return np.mean([home_dict[op_id] for op_id in self.away_opps])

    def add_home_game(self, team_score, opp_id, opp_score):
        self.home_opps.append(opp_id)
        self.home_scores.append(game_score(team_score - opp_score))

    def add_away_game(self, team_score, opp_id, opp_score):
        self.away_opps.append(opp_id)
        self.away_scores.append(game_score(team_score - opp_score))
