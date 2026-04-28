# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:47:22 2026

@author: Mira
"""
import numpy as np
import time

np.random.seed(42)

from Corrosense_9bits_random import Training_matrix
from Initializations_updated import Kmeans_plus_plus
from Final_GLA import GLA


X = Training_matrix
K = 200


# Distance counters
anchor_distance_counter = 0
bf_distance_counter = 0


def dist_anchor(a, b):
    global anchor_distance_counter
    anchor_distance_counter += 1
    return np.linalg.norm(a - b)


def dist_bf(a, b):
    global bf_distance_counter
    bf_distance_counter += 1
    return np.linalg.norm(a - b)

# Initialization + GLA
C0 = Kmeans_plus_plus(X, K)
C_final, _, A, _, _ = GLA(X, C0, 0.05, 1000)



# Build clusters
def build_sorted_clusters(X, C, A):
    K = C.shape[1]
    clusters = {}

    for i in range(K):
        idx = np.where(A[i] == 1)[0]

        if len(idx) == 0:
            clusters[i] = []
            continue

        points = X[:, idx]
        dists = np.linalg.norm(points - C[:, i].reshape(-1, 1), axis=0)

        sorted_idx = np.argsort(dists)
        clusters[i] = list(zip(idx[sorted_idx], dists[sorted_idx]))

    return clusters



# Codevector distances
def build_cluster_centroid_distances(C):
    K = C.shape[1]
    dist_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            dist_matrix[i, j] = np.linalg.norm(C[:, i] - C[:, j])

    return dist_matrix



# Anchor prunning
def anchor_pruning_fast(X, C, clusters, M=1):

    n = X.shape[1]
    K = C.shape[1]

    centroid_dist = build_cluster_centroid_distances(C)

    new_assignments = np.zeros(n, dtype=int)

    for i in range(K):

        if len(clusters[i]) == 0:
            continue

        candidate_centroids = np.argsort(centroid_dist[i])[1:M+1]

        for (idx, _) in clusters[i]:

            x = X[:, idx]

            best = i
            best_dist = dist_anchor(x, C[:, i])

            for j in candidate_centroids:

                d_x_cj = dist_anchor(x, C[:, j])

                if d_x_cj < best_dist:
                    best = j
                    best_dist = d_x_cj

            new_assignments[idx] = best

    A_new = np.zeros((K, n), dtype=int)
    A_new[new_assignments, np.arange(n)] = 1

    return A_new



# Brute force
def brute_force_all(X, C):
    n = X.shape[1]
    K = C.shape[1]

    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best = 0
        best_dist = float("inf")

        for j in range(K):
            d = dist_bf(X[:, i], C[:, j])
            if d < best_dist:
                best = j
                best_dist = d

        labels[i] = best

    return labels


def A_to_labels(A):
    return np.argmax(A, axis=0)



# Run

t0 = time.perf_counter()

clusters = build_sorted_clusters(X, C_final, A)
centroid_dist = build_cluster_centroid_distances(C_final)

t1 = time.perf_counter()

# Reset counter
anchor_distance_counter = 0

t2 = time.perf_counter()

A_anchor = anchor_pruning_fast(X, C_final, clusters)

t3 = time.perf_counter()

anchor_labels = A_to_labels(A_anchor)

# Brute force reset counter
bf_distance_counter = 0

t4 = time.perf_counter()

bf_labels = brute_force_all(X, C_final)

t5 = time.perf_counter()

#Results
print("\n--- TIMING ---")
print("Anchor build time:", t1 - t0)
print("Anchor query time:", t3 - t2)
print("Anchor total time:", t3 - t0)

print("Brute force time:", t5 - t4)

print("\n--- CORRECTNESS ---")
print("Anchor == BF:", np.mean(anchor_labels == bf_labels))

print("\n--- DISTANCE COMPARISON ---")
print("Brute force distances:", bf_distance_counter)
print("Anchor-based distances:", anchor_distance_counter)
print("Reduction (%):", 100 * (1 - anchor_distance_counter / bf_distance_counter))

