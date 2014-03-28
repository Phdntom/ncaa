
import os
from collections import defaultdict

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

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

def split_seasons(csvfname, parent_folder, basename):
    '''
    Parameters
    ----------
    csvfname:       path to csv source file name
    parent_folder:  top directory for your processed data
    basename:       used to build sub directory and filenames

    Example
    -------
    >>> split_seasons("data.csv", "top_dir", "field")
    Opens "data.csv" in current context and splits on row[0].
    Create "top_dir".
    Create directory "fieldA" where A was seen in row[0] for all A
    Open and write file contents into "fieldA.csv"
    '''


    create_dir(parent_folder)
    with open(csvfname,"r") as fobj:
        # Grab CSV file column labels
        labels = (fobj.readline()).split(',')
        # Start with no season
        cur_season = None
        # Store info about a season in a seasons[season] dict
        seasons = defaultdict(list)
        for line in fobj:
            fields = line.split(',')
            season = fields[0]
            # Just add the whole line to the appropriate list
            seasons[season].append(line)



    # Process each season and put results in parent_folder
    with change_dir(parent_folder):
        for season in seasons:
            print(season)
            # Build the dirname and create it
            dirname = basename + season
            create_dir(dirname)
            # Change to dirname and write the season file
            outname = dirname + ".csv"
            with change_dir(dirname), open(outname, "w") as outf:
                outf.write("".join(seasons[season]))


if __name__ == "__main__":

    parent_folder = "metadata"
    base_name = "season"
    season_file = "data/regular_season_results.csv"

    split_seasons(season_file, parent_folder, base_name)


        

            
