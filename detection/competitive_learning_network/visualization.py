import matplotlib.pyplot as plt
import math
import numpy as np

def network_mapping(rows, cols, nclasses, competitive_weights, linear_weights, class_label = None, figure_path = None):
  """Visualizing the competitive layer.
  
  Parameters
  ----------
  figure_path: str
    The path of file to save figure, if there is no path provided, figure will be shown
  """
  # plt.clf()

  if class_label is None:
    class_label = np.arange(0, nclasses, 1)

  # Rescaling weights to (0, 1) range
  from sklearn.preprocessing import MinMaxScaler
  sc_weights = MinMaxScaler(feature_range = (0, 1))
  weights = np.copy(competitive_weights)
  weights = sc_weights.fit_transform(weights)

  # Parameters
  n_subclass = weights.shape[0]
  n_feature = weights.shape[1]

  # Meshgrid of the layer
  meshgrid = np.zeros((rows, cols))
  for idx in range (n_subclass):
    i = rows - 1 - (idx // cols)
    j = idx % cols
    for c in range (nclasses):
      if linear_weights[c][idx] == 1:
        meshgrid[i][j] = class_label[c]
        break
  meshgrid = meshgrid.astype(np.int8)

  # Drawing meshgrid of the layer
  from matplotlib import pyplot as plt
  fig = plt.figure(figsize = (10, 10))
  global_ax = fig.add_axes([0, 0, 1, 1])
  global_ax.pcolormesh(meshgrid, edgecolors = 'black', linewidth = 0.1, alpha = 0.4)
  global_ax.set_yticklabels([])
  global_ax.set_xticklabels([])

  # Drawing for each subclass of the layer
  for idx in range (n_subclass):
    i = rows - 1 - (idx // cols)
    j = idx % cols
    cell_width = 1 / rows
    cell_height = 1 / cols

    # Name of the class, to which subclass belongs
    grid_ax = fig.add_axes([j / cols, i / rows, cell_width, cell_height], polar=False, frameon = False)
    grid_ax.axis('off')
    grid_ax.text(0.05, 0.05, int(meshgrid[i][j]))
    
    # Pie chart corresponding to weight
    polar_ax = fig.add_axes([j / cols + cell_height * 0.1, i / rows + cell_width * 0.1, cell_width * 0.8, cell_height * 0.8], polar=True, frameon = False)
    polar_ax.axis('off')
    theta = np.array([])
    width = np.array([])  
    for k in range (n_feature):
      theta = np.append(theta, k * 2 * np.pi / n_feature)
      width = np.append(width, 2 * np.pi / n_feature)
    radii = weights[idx]
    color = ['b', 'g', 'r', 'black', 'm', 'y', 'k', 'w']
    bars = polar_ax.bar(theta, radii, width=width, bottom=0.0)
    for k in range (n_feature):
      bars[k].set_facecolor(color[k])
      bars[k].set_alpha(1)
  if figure_path:
    plt.savefig(figure_path)
  else:
    plt.show()

def feature_distribution(rows, cols, weights, feature_label = None, figure_path = None):
  plt.clf()
  # n_neurons = weights.shape[0]
  n_features = weights.shape[1]
  ncols = 1
  nrows = n_features

  if feature_label is None:
    feature_label = np.arange(0, n_features, 1)
  fig, ax = plt.subplots(figsize = (1, nrows + 1))
  fig.set_size_inches((2, (nrows + 1) * 2))

  for i in range (n_features):
    plt.subplot(nrows, ncols, i + 1)
    plt.imshow(weights[:, i].reshape((rows, cols)))
    max_x = np.max(weights[:, i])
    min_x = np.min(weights[:, i])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    font_props = {'fontsize': 10, 'fontweight': 0.5}
    plt.title(str(feature_label[i]), fontdict = font_props)
    cb = plt.colorbar(use_gridspec = True, orientation = 'horizontal', ticks = [min_x, max_x], shrink = 0.75, pad = 0.05, fraction = 0.1)
    cb.outline.set_visible(False)
    cb.ax.set_xticklabels(['Low', 'High'])
  
  plt.tight_layout()
  plt.xticks([])
  plt.yticks([])
  plt.axis('off')
  if figure_path is None:
    plt.show()
  else:
    plt.savefig(figure_path)