import numpy as np
import matplotlib.pyplot as plt
from pylab import bone, plot, show, pcolor, colorbar

fig = plt.figure(figsize = (6, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.pcolormesh(np.array([[1, 2, 3, 4], [1, 4, 1, 2], [1, 2, 4, 3], [3, 4, 2, 1]]))
ax.set_yticklabels([])
ax.set_xticklabels([])

# ax = fig.add_subplot(448, projection='polar')
ax = fig.add_axes([0.25,0,0.25,0.25], polar=True)

N = 6
theta = [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
radii = [4, 5, 4, 4, 5, 6]
ii = [0, 1, 2, 3, 4, 5]
width = [np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
color=['black', 'red', 'green', 'blue', 'cyan', 'yellow']
bars = ax.bar(theta, radii, width=width, bottom=0.0)

for r,bar,i in zip(radii, bars, ii):
    bar.set_facecolor(color[i])
    bar.set_alpha(1)

# plot(
#   0.5,
#   0.5,
#   markers[0],
#   markeredgecolor = colors[0],
#   markerfacecolor = 'None',
#   markersize = 10,
#   markeredgewidth = 2
# )

ax.set_xticklabels([])
ax.set_yticklabels([])

# ax = plt.axes([0, 0, 1, 1])
# ax.set_aspect('equal')
# ax.pcolormesh(np.array([[1, 2], [3, 4]]))
# plt.yticks([0.5, 1.5], ["long long tick label",
#                         "tick label"])
# plt.ylabel("My y-label")
# plt.title("Check saved figures for their bboxes")

# savefig('../figures/polar_ex.png',dpi=48)
plt.show()