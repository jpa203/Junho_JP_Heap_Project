import matplotlib.pyplot as plt
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')
from collections import defaultdict
from dataclasses import dataclass
from nltk.corpus import stopwords
import os
from scipy.spatial.distance import cosine
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from read_docx_tf_idf import read_docx_tf_idf

def draw_heap(heap,counter=0):
    G = nx.DiGraph()
    # add nodes to the graph
    for i,val in enumerate(heap):
        G.add_node(i)
    
    # set labels of nodes to the corresponding file names
    labels = {i: val[3].replace('.docx', '') for i, val in enumerate(heap)}
    
    for i in range(len(heap)):
        if 2*i+1 < len(heap):
            G.add_edge(i, 2*i+1)
        if 2*i+2 < len(heap):
            G.add_edge(i, 2*i+2)
    
    # Set the node size and position for a pyramid-like visualization
    pos = {}
    node_size = []
    for i in range(len(heap)):
        depth = len(bin(i+1))-3  # calculate the depth of the node in the heap
        x_pos = i - 2**(depth-1)  # calculate the x-position of the node
        y_pos = -depth  # calculate the y-position of the node
        pos[i] = (x_pos, y_pos)
        node_size.append(1000/(2**(depth-1)))
            
    # Draw the heap
    nx.draw_networkx(G, pos=pos, node_size=node_size, font_size=12, node_color='w', edgecolors='k', linewidths=1, labels=labels)
    plt.savefig(f'heap_{counter}.png')
    plt.show()
    counter += 1



def draw_best_heap(heap,counter=0):
    G = nx.DiGraph()
    # add nodes to the graph
    for i,val in enumerate(heap):
        G.add_node(i)
    
    # set labels of nodes to the corresponding file names
    labels = {i: val[3].replace('.docx', '') for i, val in enumerate(heap)}
    
    for i in range(len(heap)):
        if 2*i+1 < len(heap):
            G.add_edge(i, 2*i+1)
        if 2*i+2 < len(heap):
            G.add_edge(i, 2*i+2)
    
    # Set the node size and position for a pyramid-like visualization
    pos = {}
    node_size = []
    for i in range(len(heap)):
        depth = len(bin(i+1))-3  # calculate the depth of the node in the heap
        x_pos = i - 2**(depth-1)  # calculate the x-position of the node
        y_pos = -depth  # calculate the y-position of the node
        pos[i] = (x_pos, y_pos)
        node_size.append(1000/(2**(depth-1)))
            
    # Draw the heap
    nx.draw_networkx(G, pos=pos, node_size=node_size, font_size=12, node_color='w', edgecolors='k', linewidths=1, labels=labels)
    plt.title("Best_heap_Structure")
    plt.savefig(f'best_heap.png')
    plt.show();
