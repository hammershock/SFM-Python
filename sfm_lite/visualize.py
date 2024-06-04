import random

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkAgg')


def visualize_edges(G):
    for u, v in G.edges:
        # draw visualization
        visualize_edge(G, u, v)


def visualize_edge(G, u, v):
    edge = G[u][v]
    image_show = cv2.drawMatches(G.nodes[u]['image'], G.nodes[u]['kps'], G.nodes[v]['image'], G.nodes[v]['kps'], edge['matches'], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image_show = cv2.resize(image_show, None, None, fx=0.4, fy=0.4)
    cv2.imshow('', image_show)
    cv2.waitKey(0)


# Initialize color map dictionary if color indices are provided
color_map = {}


def visualize_points3d(points3d, colors=None, color_indices=None, s=10):
    points3d = np.array(points3d)
    assert points3d.shape[1] == 3, "Input should be a Nx3 numpy array"
    if colors is not None:
        colors = np.array(colors)
        assert colors.shape == points3d.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if color_indices is not None and colors is None:
        unique_indices = np.unique(color_indices)
        for index in unique_indices:
            if index not in color_map:
                # Assign a random color for new indices
                color_map[index] = [random.random() for _ in range(3)]
        # Scatter plot using color indices
        for idx in unique_indices:
            idx_mask = color_indices == idx
            ax.scatter(points3d[idx_mask, 0], points3d[idx_mask, 1], points3d[idx_mask, 2],
                       color=color_map[idx], label=f"Index {idx}", s=s)
    elif colors is not None:
        # Convert BGR to RGB
        colors_rgb = colors[:, [2, 1, 0]] / 255.
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], color=colors_rgb, s=s)
    else:
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=s)

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Title and legend
    ax.set_title('3D Point Visualization')

    plt.show()


def visualize_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='#909090', node_size=2000, arrowstyle='-|>', arrowsize=20)

    plt.title("Co-visibility Graph Visualization")
    plt.show()
