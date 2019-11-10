import numpy as np
import graph_tool as gt
from tensorflow.keras.layers import Dense

def dense_model_to_graph(model):
    input_layer = model.layers[0]
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    d = len(dense_layers)
    h = [layer.units for layer in dense_layers]
    Ws = [layer.get_weights()[0] for layer in dense_layers]
    N = Ws[0].shape[0] + sum(W.shape[1] for W in Ws)
    edges = []
    i_in = 0
    for W in Ws:
        N_in, N_out = W.shape
        i_out = i_in + N_in
        edges += [(i, j) for i in range(i_in, i_out) for j in range(i_out, i_out + N_out)]
        i_in = i_out
    g = gt.Graph(directed=False)
    g.add_edge_list(edges)
    weights = np.concatenate([W.flatten() for W in Ws])
    g.ep['weight'] = g.new_edge_property('float', weights)
    return g
