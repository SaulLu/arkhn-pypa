import glob
import os

def get_path_last_model():
    list_of_files = glob.glob('data/parameters/intermediate/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return(latest_file)

if __name__ == "__main__":
    print(get_path_last_model())