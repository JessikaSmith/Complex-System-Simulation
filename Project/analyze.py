import networkx as nx
import matplotlib.pyplot as plt
from processing import *

def plot_degree_distribution():
    pass

def create_graph(data):
    G = nx.Graph()
    for i, j in data.values:
        G.add_edge(i, j)
    return G

def main():
    fname = 'Datasets/Cit-HepPh.txt'
    data = read_txt(fname)
    graph = create_graph(data)
    print(graph.edges())

if __name__ == '__main__':
    main()