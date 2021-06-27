import numpy as np
import matplotlib.pyplot as plt
import json


def angle_dist(a, b):
    clockwise_angle = np.abs(b-a)
    anticlockwise_angle = np.abs( min(a,b) + (np.pi*2 - max(a,b)) )
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
    powr = np.linspace(1,3)
    steps = int(np.pi*2/theta)
    desired_acc = theta/(np.pi*2)
    weights = []
    for j in range(powr.shape[0]):
        weights.append(isol_arr(data, P=powr[j]))

    err_mat = np.zeros((steps,powr.shape[0]))
    for i in range(0,steps):
        scores = np.array([1 if i*theta <= d <= (i+1)*theta else 0.0 for d in data])
        print("SHAPE")
        print(powr.shape[0])
        for j in range(powr.shape[0]):
            weighted_score = np.dot(weights[j], scores)
            err_mat[i,j] = np.abs(weighted_score - desired_acc)
    print("BEST POWER")
    print(powr)
    err_vals = np.sum(err_mat, axis=0) / steps
    print(err_vals)
    plt.plot(powr, err_vals)
    plt.savefig("best-power.png")


data = [np.deg2rad(x) for x in json.load(open("az_data.txt"))["val"]]

p = np.linspace(1,4,21)

for i in range(21):
    print("Creating weights for p={}...".format(np.round(p[i])))
    isol = isol_arr(data, p[i])
    np.save(
        open("weights/weight-val-{}.npy".format(np.round(p[i],1)), "wb"),
        isol
    )

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