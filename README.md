ncaa
====

split_seasons.py
----------------
Minimal script to quickly split the seasons.csv into multiple seasonX.csv in a directory structure.

season_accumulator.py
---------------------
Gathers stats for each seasonX.csv and builds a Rating Matrix in the collaborative filtering sense.

Using the netflix movie rating paradigm as an example, each user has rated a collection of movies.
The matrix is N_users x N_movies and is in general not square and is missing many entries (i.e. all users
have not rated all movies).

By analogy, for sports contents, there are N_teams and N_opponents (in this case N_teams==N_opponents) where
each N_team rates a subset in N_opponents. In this case, the matrix is square but still has many missing entries,
that is to say, all teams have not played all opponents.

Metric: game_score and game_rating
Uses a chessrank style system where ratings are awarded based on point differential.

Outputs: YmatrixX.npy
Each "rating" matrix is written as a file in the directory structure used by split_seaons.py.

matrix_factorization.py
-----------------------
Input:    Rating Matrix Y
Ouput:    P, Q matrix which optimally factorize Y given K latent features.



main.py
-------
Calls each module

Loads all YmatrxX.npy matrices in the directory structure and procudes a completed rating matrix by finding
an optimal factorization using N_latent features as the inner product dimension.

Outputs: eYmatrix.npy
