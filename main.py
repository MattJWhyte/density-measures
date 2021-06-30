import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import random


def angle_dist(a, b):
    a = np.rad2deg(a)
    b = np.rad2deg(b)
    clockwise_angle = np.abs(b-a)
    anticlockwise_angle = np.abs( min(a,b) + (360.0 - max(a,b)) )
    return min(clockwise_angle,anticlockwise_angle)


def isolation_measure(a, data, P):
    return sum([angle_dist(a, d)**P for d in data])**P/len(data)


def isol_arr(data, P):
    n = len(data)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = angle_dist(data[i], data[j])**P
            dist[j, i] = angle_dist(data[i], data[j])**P
    sum_i = np.sum(dist)
    return np.sum(dist, axis=1) / sum_i


# Finds best
def get_best_pow(theta, data):
    powr = np.linspace(1,4,21)
    steps = int(np.pi*2/theta)
    desired_acc = theta/(np.pi*2)
    weights = []
    for j in range(powr.shape[0]):
        weights.append(np.load(open("weights/weight-val-{}.npy".format(np.round(powr[j],1)), "rb")))

    err_mat = np.zeros((steps,powr.shape[0]))
    for i in range(0,steps):
        scores = np.array([1 if i*theta <= d <= (i+1)*theta else 0.0 for d in data])
        for j in range(powr.shape[0]):
            weighted_score = np.dot(weights[j], scores)
            err_mat[i,j] = np.abs(weighted_score - desired_acc)

    print("BEST POWER")
    print(powr)
    err_vals = np.sum(err_mat, axis=0) / steps
    plt.plot(powr, err_vals)
    plt.savefig("best-power.png")
    plt.clf()
    return err_vals


def save_weights(data, p=np.linspace(1,4,21)):
    for i in range(p.shape[0]):
        print("Creating weights for p={}...".format(np.round(p[i])))
        isol = isol_arr(data, p[i])
        np.save(
            open("weights/weight-val-{}.npy".format(np.round(p[i], 1)), "wb"),
            isol
        )


data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"]]

save_weights(data, np.array([2.0]))

'''
data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"]]
powr = np.linspace(1, 4, 21)

err_vals = np.zeros((21,))
for i in range(1,5):
    err_vals += get_best_pow(np.pi/i, data)

plt.plot(powr, err_vals/4.0)
plt.savefig("total-best-power.png")'''


def get_bins(weights, data):
    weighted_bins = []
    for i in range(24):
        s = sum([weights[j] if np.deg2rad(15 * i) <= d < np.deg2rad(15 * (i + 1)) else 0.0 for j, d in enumerate(data)])
        weighted_bins.append(s)
    return weighted_bins


w = np.load(open("weights/weight-val-2.0.npy", "rb"))
bins = get_bins(w, data)

f = plt.figure()
ax = f.add_subplot(projection="polar")
new_bins = get_bins([1.0/len(data) for _ in range(len(data))], data)
ax.plot([np.pi*2/24.0*i+np.pi/24.0 for i in range(24)] + [np.pi/24.0], new_bins + [new_bins[0]])
ax.plot([np.pi*2/24.0*i+np.pi/24.0 for i in range(24)] + [np.pi/24.0], bins + [bins[0]])
plt.savefig("test.png")

sys.exit()

min_b = min(bins)
max_b = max(bins)

plt.plot([i*np.pi/12+np.pi/24 for i in range(0,24)], bins)

ct = 0

for i in range(24):
    print("Slice {}".format(i))
    while bins[i] > 1.1*min_b:
        idx = []
        for j,d in enumerate(data):
            if np.deg2rad(15 * i) <= d < np.deg2rad(15 * (i + 1)):
                idx.append(j)
        #r_idx = idx[random.randint(0, len(idx))]
        w[idx] = w[idx]*0.95
        bins = get_bins(w, data)

plt.plot([i*np.pi/12+np.pi/24 for i in range(0,24)], bins)
plt.show()

np.save(open("weights/pascal3d-val-weights.npy", "wb"), w)

'''
data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"][:4000]]

n = len(data)

scores = [0.0 if d < np.pi/2 or d > np.pi*3/2 else 1.0 for d in data]

ct = 0
for d in data:
    if d < np.pi / 2 or d > np.pi * 3 / 2:
        ct += 1

P = np.e

# Vary
# Sector size (30 - 60 - 90 - 120 - 180)
# Position (12 - 6 - 4 - 3 - 2)
# P-value (1.5, 2.0, 2.5, 3.0, 3.5)


f = plt.figure()
ax = f.add_subplot(projection="polar")

i = [isolation_measure(a, data) for a in data]
sum_i = sum(i)
i = [el/sum_i for el in i]

ax.hist(data)
plt.show()

print("Weighted score")
print(sum([i[j]*scores[j] for j in range(0,len(data))]))
print("Not weighted score")
print(sum([1*scores[j] for j in range(0,len(data))])/4000)
'''