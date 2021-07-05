import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import random
import matplotlib.cm as cm


def distance_elevation_azimuth(xyz):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    theta = np.abs(90-np.rad2deg(np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))))
    c = z < 0
    theta = theta - 2*c*theta
    #if z < 0:
    #    theta *= -1.0
    phi = np.rad2deg(np.arctan2(y,x))
    d = phi < 0
    phi = phi + 360.0*d
    #if phi < 0.0:
    #    phi += 360.0
    return [np.sqrt(x**2+y**2+z**2), theta, phi]


def distance_elevation_azimuth_old(xyz):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    theta = np.abs(90-np.rad2deg(np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))))
    if z < 0:
        theta *= -1.0
    phi = np.rad2deg(np.arctan2(y,x))
    if phi < 0.0:
        phi += 360.0
    return [np.sqrt(x**2+y**2+z**2), theta, phi]


def angle_dist(a, b, deg=False):
    if deg:
        a = np.rad2deg(a)
        b = np.rad2deg(b)
    clockwise_angle = np.abs(b-a)
    anticlockwise_angle = np.abs( min(a,b) + (360.0 - max(a,b)) ) if deg else np.abs( min(a,b) + (np.pi*2.0 - max(a,b)) )
    return min(clockwise_angle,anticlockwise_angle)


def isolation_measure(a, data, P):
    return sum([angle_dist(a, d)**P for d in data])**P/len(data)


def angle_dist_mat(data, P):
    n = len(data)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = angle_dist(data[i], data[j]) ** P
            dist[j, i] = angle_dist(data[i], data[j]) ** P
    return dist


def isol_arr(data, P, deg=False):
    dist = angle_dist_mat(data, P)
    sum_i = np.sum(dist)
    return np.sum(dist, axis=1) / sum_i


def isol_arr_new(data, P, deg=False):
    n = len(data)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = angle_dist(data[i], data[j], deg=deg)**P
            dist[j, i] = angle_dist(data[i], data[j], deg=deg)**P
    dist = np.sort(dist, axis=1) @ np.array([1.0**i for i in range(n)])
    return dist / np.sum(dist)


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


def save_weights(data, dset="val", p=np.linspace(1,4,21), deg=False):
    for i in range(p.shape[0]):
        print("Creating weights for p={}...".format(np.round(p[i])))
        isol = isol_arr(data, p[i], deg=deg)
        np.save(
            open("weights/weight-{}-{}-{}.npy".format(dset, np.round(p[i], 1), "deg" if deg else "rad"), "wb"),
            isol
        )


def discounted_distance(d):
    return np.exp((-6.0/np.pi)*(d**2))


def intersection(I1, I2):
    if I1[0] > I1[1]:
        I1[0] = I1[0]-360.0
    if I2[0] > I2[1]:
        I2[0] = I2[0]-360.0
    m = max(I1[0], I2[0])
    M = min(I1[1], I2[1])
    return max(M-m, 0)


def clockwise_len(I):
    if I[0] > I[1]:
        return I[1] - (I[0]-360.0)
    return I[1]-I[0]


# Data must be in degrees
def create_samples(deg_data, dir):
    n = len(deg_data)
    annotation = np.zeros(46656)
    ct = 0
    for i in range(0,360,10):
        if i % 40 == 0:
            print("{}% complete...".format(np.round(ct/46646.0*100.0)))
        for i_s in range(15, 100, 15):
            for j in range(0,360,10):
                for j_s in range(15, 100, 15):
                    i_int = [(i - i_s) % 360, (i + i_s) % 360]
                    j_int = [(j - j_s) % 360, (j + j_s) % 360]
                    intersect = intersection(i_int, j_int)
                    acc = np.zeros(n)
                    i_rand = np.random.rand()*0.25
                    j_rand = np.random.rand()*0.25
                    rest_rand = np.random.rand()*0.25+0.75
                    for k,d in enumerate(deg_data):
                        if i_int[0] <= d <= i_int[1]:
                            acc[k] = 1.0 if np.random.rand() > i_rand else 0.0
                        elif j_int[0] <= d <= j_int[1]:
                            acc[k] = 1.0 if np.random.rand() > j_rand else 0.0
                        else:
                            acc[k] = 1.0 if np.random.rand() > rest_rand else 0.0
                    annotation[ct] = (clockwise_len(i_int)*(1-i_rand) + (clockwise_len(j_int)-intersect)*(1-j_rand)
                           + (360.0-(clockwise_len(i_int)+clockwise_len(j_int)-intersect))*(1-rest_rand))/360.0
                    # annotation[ct] = (clockwise_len(i_int)+clockwise_len(j_int)-intersect)/360.0
                    np.save(dir+"sample-{}.npy".format(ct), acc)
                    ct += 1
    np.save(dir + "annotation.npy", annotation)


val_deg_data = [x for x in json.load(open("az_data.txt"))["val"]]
create_samples(val_deg_data, "datasets/val_dataset/")


sys.exit()
val_data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"]]
#train_data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["train"]]

dist_mat = np.load("val_abs_dist_mat.npy")
discounted_dist_mat = np.frompyfunc(discounted_distance, 1, 1)
dist_mat = discounted_dist_mat(dist_mat)
dist_mat = np.sum(dist_mat, axis=1)**(-1)
dist_mat = dist_mat/np.sum(dist_mat)

print(np.sum(dist_mat))


f = plt.figure()
ax = f.add_subplot(1,1,1,projection="polar")
ax.scatter(val_data, dist_mat, s=5)
plt.savefig("new-weighted-dist-3.png")

np.save("rbf-weights-val.npy", dist_mat)

#save_weights(val_data, "val", np.array([2.0]))
#save_weights(val_data, "train", np.array([2.0]))



sys.exit()

'''
data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"]]
powr = np.linspace(1, 4, 21)

err_vals = np.zeros((21,))
for i in range(1,5):
    err_vals += get_best_pow(np.pi/i, data)

plt.plot(powr, err_vals/4.0)
plt.savefig("total-best-power.png")'''


def get_bins(data, weights=None):
    if weights is None:
        weights = [1.0/len(data) for _ in range(len(data))]
    weighted_bins = []
    for i in range(24):
        s = sum([weights[j] if np.deg2rad(15 * i) <= d < np.deg2rad(15 * (i + 1)) else 0.0 for j, d in enumerate(data)])
        weighted_bins.append(s)
    return weighted_bins


w = isol_arr_new(data, 2.0)
weighted_bins = get_bins(data, w)
bins = get_bins(data)
f = plt.figure()
ax = f.add_subplot(1,2,1,projection="polar")
ax.bar([np.deg2rad(15.0*(i+0.5)) for i in range(24)], weighted_bins, width=np.deg2rad(15.0), edgecolor='k')
ax2 = f.add_subplot(1,2,2,projection="polar")
ax2.bar([np.deg2rad(15.0*(i+0.5)) for i in range(24)], bins, width=np.deg2rad(15.0), edgecolor='k')
print(bins)
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