import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import cm
import random

QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '1':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

NOTES = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}

QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]}

NOTES_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G","G#","A","A#","B"]


def generate_chord_dict():
    dict = {}
    qualities = ["maj", "min", "7"]
    for i in range(12):
        for k in range(3):
            q = qualities[k]
            label = NOTES_LIST[i]+":"+q
            chroma = rotate(QUALITIES[q],-i)
            dict[label] = chroma
    return dict


def rotate(l, n):
    return l[n:] + l[:n]

CHORDS = generate_chord_dict()

def chroma_to_chord(chroma):
  chord = ""
  for k in range(12):
    if chroma[k] > 0:
      chord+=NOTES_LIST[k]+" "
  return chord

def transpose(chord, semitones):
    return rotate(chord, -semitones)

def get_tonic(chroma):
    """
    returns the tonic of a chord chroma
    :param chroma:
    :return:
    """

def chord_distance(chord1, chord2):
    """
    :param c1: chord name (str)
    :param c2: chord name (str)
    :return: float representing the score between c1 and c2
    """
    c1_tonic = chord1[0]
    c2_tonic = chord2[0]
    c1_mode = chord1[1:]
    c2_mode = chord2[1:]
    c1_tonic_index = NOTES_LIST.index(c1_tonic)
    c2_tonic_index = NOTES_LIST.index(c2_tonic)
    c1 = CHORDS[chord1]
    c2 = CHORDS[chord2]
    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    if (NOTES_LIST[(c1_tonic_index+7)%12]) == c2_tonic: #si c2 est la dominante de c1
        if c2_mode != "min":
            dist = min(1, dist)
    if (NOTES_LIST[(c2_tonic_index + 7) % 12]) == c1_tonic:  # si c1 est la dominante de c2
        if c1_mode != "min":
            dist = min(1, dist)
    return dist/2.8284271247461903 #normalization to [0,1]

def rotate(l, n):
    return l[n:] + l[:n]

def chroma_to_chord(chroma):
  chord = ""
  for k in range(12):
    if chroma[k] > 0:
      chord+=NOTES[k]+" "
  return chord

def generate_dataset(N):
    dataset = []
    qualities = ["maj", "min"]
    for i in range(N):
        k = random.randrange(0, 12)  # fondamentale de l'accord
        q = random.choice(qualities)
        label = NOTES[k] + ":" + q
        chroma = np.array(rotate(QUALITIES[q], -k)).reshape(12, 1, 1)
        chroma = chroma - np.random.rand(12, 1, 1) / 2
        dataset.append([chroma, label])

    return pd.DataFrame(dataset, columns=["Chroma", "Label"])


synthetic_df = generate_dataset(10000)
test_slices = []
test_labels = []
for i, row in synthetic_df.iterrows():
    test_slices.append(row["Chroma"])
    test_labels.append(row["Label"])

test_slices = np.array(test_slices)
test_slices.shape

z_test = encoder.predict(test_slices, batch_size=128)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(z_test[2])

x,y = principalComponents[:,0], principalComponents[:,1]
labels = test_labels

label_coords = {}

for i in range(len(x)):
  if labels[i] in label_coords:
    label_coords[labels[i]].append((x[i],y[i]))
  else:
    label_coords[labels[i]] = [(x[i],y[i])]

label_centroids = {}
for l in label_coords.keys():
  label_centroids[l] = centroid(label_coords[l])


c_points = list(label_centroids.values())
c_labels = list(label_centroids.keys())


cmap = cm.get_cmap('viridis', 1000)

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



chords = c_labels
points = np.array(c_points)

#chords[k] <-> points[k]
K=5
nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)
scores = np.zeros(len(points))
for k in range(len(points)):
       neighbors = indices[k][1:]
       score = 0
       c1 = chords[k]
       for i in neighbors:
              c2 = chords[i]
              d = chord_distance(c1, c2)
              score+=d
       scores[k] = score/K

scores = scores
# compute Voronoi tesselation
vor = Voronoi(points)
fig, ax = plt.subplots()
ax.set_title("Score : {:.2f}".format(sum(scores)))
# plot
regions, vertices = voronoi_finite_polygons_2d(vor)
fig.set_size_inches(18.5, 10.5)
# colorize
for i, region in enumerate(regions):
    polygon = vertices[region]
    c = cmap(scores[i])
    ax.fill(*zip(*polygon), color=c)

ax.plot(points[:,0], points[:,1], 'ko')
voronoi_plot_2d(vor, ax, show_points=False,show_vertices=False)
for k in range(len(points)):
       x, y = points[k][0], points[k][1]
       ax.annotate(chords[k]+"\n"+str(truncate(scores[k],2)),(x,y))
plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

plt.show()

#reconstruction
test_index=random.randrange(0,1000)
z_test = encoder.predict(slices_train[0:1000], batch_size=128)

output = decoder.predict(z_test[0])[test_index].squeeze(axis=2)

#f = plt.gcf()
#f.set_size_inches(20, 10)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].imshow(output,origin='lower',cmap="Greens")
axes[1].imshow(slices_train[test_index].squeeze(axis=2),origin='lower',cmap="Reds")
fig.tight_layout()
