import numpy as np
import graph_tool as gt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense

def dense_model_to_graph(model):
    input_layer = model.layers[0]
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    d = len(dense_layers)
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

def dense_activations_to_graph(model, x_in, thresh=1.0E-5):
    G = dense_model_to_graph(model)
    input_layer = model.layers[0]
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    dense_outputs = [layer.output for layer in dense_layers]
    dense_func = K.function(inputs=model.inputs, outputs=dense_outputs)
    outputs = dense_func(x_in)
    masks = [np.expand_dims(output > thresh, axis=1) for output in outputs]
    Ws = [np.expand_dims(layer.get_weights()[0], axis=0) for layer in dense_layers]
    Ws_masked = np.concatenate([(W*mask).reshape((-1, np.prod(W.shape[1:]))) for W, mask in zip(Ws, masks)], axis=1)
    edge_masks = ~np.isclose(Ws_masked, 0.0, atol=thresh)
    Gs = [gt.GraphView(G, efilt=G.new_ep('bool', vals=mask)) for mask in edge_masks]
    return G, Gs
