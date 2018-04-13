from scene_tools import *

import matplotlib
import matplotlib.pyplot as plt

with open('stimuli/spring/1519497671_metropolis.pkl', 'rb') as f:
    dt = pickle.load(f)

if False:
    rest_length_buckets = {}
    stiffness_buckets = {}
    damping_buckets = {}
    for i in range(30, 230, 20):
        rest_length_buckets[i] = 0
    for i in range(20, 200, 20):
        stiffness_buckets[i] = 0
    for i in range(0, 9, 1):
        damping_buckets[i] = 0

    for i in dt:
        rounded = (i[0] - 30) // 20 * 20 + 30
        if rounded in rest_length_buckets:
            rest_length_buckets[rounded] += 1
        rounded = (i[1] - 20) // 20 * 20 + 20
        if rounded in stiffness_buckets:
            stiffness_buckets[rounded] +=1
        rounded = int((i[2]) * 10)
        if rounded in damping_buckets:
            damping_buckets[rounded] +=1
        else:
            print('{} not in buckets!'.format(rounded))

    print(rest_length_buckets)
    print(stiffness_buckets)
    print(damping_buckets)

xys = [[],[]]
for i in dt:
    xys[0].append(i[0])
    xys[1].append(i[1])


fig = plt.figure()
plt.title('density')
plt.xlabel('rest_length')
plt.ylabel('stiffness')
nbins = 100
#plt.scatter(xys[0], xys[1], marker='o')
plt.hexbin(xys[0], xys[1], gridsize=nbins, cmap=plt.cm.BuGn_r)
#plt.hist2d(xys[0], xys[1], bins=nbins, cmap=plt.cm.BuGn_r)


plt.show()
# fig_name = os.path.join(plots_folder, '{}_real.png'.format(now))
# print('Saved ' + fig_name)
# plt.savefig(fig_name)