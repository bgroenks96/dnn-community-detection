import numpy as np
import graph_tool as gt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from functools import reduce

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
    # initialize keras op that includes all layer outputs
    dense_func = K.function(inputs=model.inputs, outputs=dense_outputs)
    # apply function to input
    layer_outputs = dense_func(x_in)
    # add axis to all outputs (including input layer)
    outputs = [np.expand_dims(output, axis=-1) for output in [x_in] + layer_outputs]
    # compute output masks
    masks = [output > thresh for output in outputs]
    # for each layer, starting with the input, broadcast the output of shape (batch, d, 1) to (batch, d, k)
    # where k is the dimension of the next layer; i.e. repeat outputs for each node
    Ws = [np.broadcast_to(output, (output.shape[0], output.shape[1], layer.get_weights()[0].shape[-1])) \
          for layer, output in zip(dense_layers, outputs)]
    # 1. apply masks to outputs
    # 2. reshape/flatten masks from (batch, d, k) to (batch, d*k)
    def apply_mask(W, mask):
        return (W*mask).reshape((-1, np.prod(W.shape[1:])))
    # 3. concatenate all masked outputs together to get full edge list
    Ws_masked = np.concatenate([apply_mask(W, mask) for W, mask in zip(Ws, masks)], axis=1)
    # create mask over final outputs for graph-tool edge filter
    edge_masks = ~np.isclose(Ws_masked, 0.0, atol=thresh)
    # initialize graph views for all outputs in batch
    Gs = [gt.GraphView(G, efilt=G.new_ep('bool', vals=mask)) for mask in edge_masks]
    layer_sizes = [x_in.shape[-1]] + [layer.units for layer in dense_layers]
    labels = reduce(lambda cum, n: cum + n, [[i]*size for i, size in enumerate(layer_sizes)])
    # add degree and layer properties to graph
    for g, acts in zip(Gs, Ws_masked):
        g.ep['activation'] = g.new_ep('float', vals=acts)
        g.vp['degree'] = g.degree_property_map('total', weight=g.ep['activation'])
        g.vp['layer'] = g.new_vp('int', labels)
    return G, Gs