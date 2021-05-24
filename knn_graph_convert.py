"""
this program introduces knn-graph technique and node2vec method to traditional machine learning.

"""
pip install node2vec
from node2vec import node2vec
class Node2Vec(Node2Vec):
  """
  Parameters
  ----------
  p : float
      p parameter of node2vec
  q : float
      q parameter of node2vec
  d : int
      dimensionality of the embedding vectors
  """
  def __init__(self, graph, p=1, q=1, d=32):
    super().__init__(
                     graph = graph,
                     walk_length=10,
                     p=p,
                     q=q,
                     dimensions =d
                  )

import csv
import pandas as pd
import numpy as np 
import networkx as nx
from sklean.neighbors import kneighbors_graph

def main():
    # import original dataset
    FILE_NAME = '/content/NBI2020.txt'
    df = get_data_vector(FILE_NAME)
    
    # encoding original dataset with one_hot_encoder method and label_encoder method
    df_encoded = encode_data(df)
    
    # convert vectors to a graph
    vectors = df_encoded.iloc[:, 0:39]  # selected features
    n = 4  # the number of neighbors
    G = vectors_to_graph(vectors, n)

    # embedding nodes in graph into vectors
    embedded_vectors = graph_to_vectors(G,1,1,8)
    np.savetxt('result.csv', embedded_vectors,delimiter=',')  # write embedded vectors into a csv file


def write_result():
    with open('result.csv','w') as f:



def graph_to_vectors(G,p,q,d):
    """
    convert a graph into vectors with the node2vec method
    input: a graph G, p, q, and dimention d
    output: vectors with d dimentions
    """
    n2v_model = Node2Vec(G, p, q, d)  # Fit embedding model to the G graph
    model = n2v_model.fit(window=10, min_count=1, batch_words=4)  # 
    embedded_vectors =  model.wv.vectors  # Node to vec representation
    return embedded_vectors


def vectors_to_graph(vectors, n):
    """
    vert a vector_based dataset to a graph by using KNN method
    ut: vectors of selected features, the number of neighbors N.
    put: a graph G
    """
    A = kneighbors_graph(vectors, n_neighbors=n)  # generate adjacency matrix
    G = nx.Graph(A)  # generate knn graph with networkx
    return G


 # encode four nominal variables with one_hot_code
def encode_data(df):
    # state_code_001, OWNER_022, FUNCTIONAL_CLASS_026, SERVICE_UND_042B
    df_temp = pd.get_dummies(df, columns=['STATE_CODE_001','OWNER_022','FUNCTIONAL_CLASS_026','SERVICE_UND_042B'], 
                            prefix='one_hot_code', drop_first=True)

    # rearrange columns
    cols_at_end = ['BRIDGE_CONDITION', 'SUPERSTRUCTURE_COND_059', 'YEAR_BUILT_027']
    df_temp = df_temp[[c for c in df_temp if c not in cols_at_end] 
            + [c for c in cols_at_end if c in df_temp]]

    # encode BRIDGE_CONDITION
    """
    BRIDGE_CONDITION
    using replace function to encode labels
    'G, F, P' is encoded as '0, 1, 2'.
    """
    df_temp = df_temp.replace({'BRIDGE_CONDITION': {'G':1, 
                                                    'F':2,
                                                    'P':3}})
    return df_temp   


# import NBI2020.txt which has been preprocessed with matlab
def get_data_vector(file_name):
    df = pd.read_csv(file_name,sep=',', header=0)
    return df

if __name__ == '__main__':
    main()