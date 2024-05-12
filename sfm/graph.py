

def mark_edge_constructed(G, edge, i, j, pos3d_index):
    """
    标记这条边中i-j匹配点所在的track已经被重建过
    """
    u, v = edge
    edge_data = G[u][v]
    for node_idx, feat_idx in edge_data["tracks"][(i, j)]:
        G.nodes[node_idx]["constructed"][feat_idx] = pos3d_index
