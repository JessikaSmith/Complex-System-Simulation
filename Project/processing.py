import pandas as pd


def save_to_csv(data):
    data.rename(columns={0: "Source", 1: "Target"}, inplace=True)
    data.to_csv('Project/Cit-HepPh.csv', header=True, index=False)


def read_txt(fname):
    data = pd.read_csv(fname, sep="\t", header=None)
    # save_to_csv(data)
    return data

def read_csv(fname):
    data = pd.read_csv(fname)
    return data

def main():
    fname = 'Project/Cit-HepPh.txt'
    data = read_txt(fname)
    return


if __name__ == "__main__":
    main()
