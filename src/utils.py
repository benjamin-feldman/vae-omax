import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
    '': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

NOTES = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}

NOTES_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G","G#","A","A#","B"]

def rotate(l, n):
    return l[n:] + l[:n]
def generate_chord_dict():
    """
    Generate a dictionary of chord labels and their corresponding chroma slices.

    Returns:
    dict: A dictionary where the keys are chord labels in the format "note:quality" and the values are the corresponding chroma slices.
    """
    chord_dict = {}
    qualities = ["maj", "min", "7"]
    for i in range(12):
        for k in range(3):
            q = qualities[k]
            label = NOTES_LIST[i]+":"+q
            chroma = rotate(QUALITIES[q],-i)
            chord_dict[label] = chroma
    return chord_dict

CHORDS = generate_chord_dict()

def enrich_chroma(c):
    """
    Enrich a chroma slice by adding ghost harmonics.

    Args:
    c (numpy.ndarray): The input chroma slice.

    Returns:
    numpy.ndarray: The enriched chroma slice.
    """
    c_squeezed = np.copy(c.squeeze(-1).squeeze(-1))
    ghost_harmonics = [7, 4, 12]
    to_add = []
    for i in ghost_harmonics:
        c_harmonics = 0.5*random.random()*np.roll(c_squeezed, i)
        to_add.append(c_harmonics)
    for h in to_add:
        c_squeezed += h
    c_squeezed = c_squeezed/np.max(c_squeezed)
    return c_squeezed.reshape(12,1,1)

def centroid(points):
    """
    Calculate the centroid of a set of points.

    Args:
    points (list): A list of points in the form [(x1, y1), (x2, y2), ...].

    Returns:
    list: The coordinates of the centroid in the form [x, y].
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]

def chroma_to_chord(chroma):
    """
    Convert a chroma slice to a chord name.

    Args:
    chroma (numpy.ndarray): The input chroma slice.

    Returns:
    str: The chord name.
    """
    chord = ""
    for k in range(12):
        if chroma[k] > 0:
            chord += NOTES_LIST[k] + " "
    return chord

def transpose(chord, semitones):
    """
    Transpose a chord by a given number of semitones.

    Args:
    chord (str): The input chord name.
    semitones (int): The number of semitones to transpose the chord.

    Returns:
    str: The transposed chord name.
    """
    return rotate(chord, -semitones)

def chord_distance(chord1, chord2):
    """
    Calculate the distance between two chords.

    Args:
    chord1 (str): The first chord name.
    chord2 (str): The second chord name.

    Returns:
    float: The distance between the two chords.
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
    if (NOTES_LIST[(c1_tonic_index+7)%12]) == c2_tonic: # if c2 is the dominant of c1
        if c2_mode != "min":
            dist = min(1, dist)
    if (NOTES_LIST[(c2_tonic_index + 7) % 12]) == c1_tonic:  # if c1 is the dominant of c2
        if c1_mode != "min":
            dist = min(1, dist)
    return dist/2.8284271247461903 # normalization to [0,1]

def chroma_to_chord(chroma):
    """
    Convert a chroma slice to a chord name.

    Args:
    chroma (numpy.ndarray): The input chroma slice.

    Returns:
    str: The chord name.
    """
    chord = ""
    for k in range(12):
        if chroma[k] > 0:
            chord += NOTES[k] + " "
    return chord

def generate_dataset(N):
    """
    Generate a synthetic dataset of chroma slices and corresponding chord labels.

    Args:
    N (int): The number of samples to generate.

    Returns:
    pandas.DataFrame: The synthetic dataset with columns "Chroma" and "Label".
    """
    dataset = []
    qualities = ["maj", "min"]
    for i in range(N):
        k = random.randrange(0, 12)  # fundamental of the chord
        q = random.choice(qualities)
        label = NOTES[k] + ":" + q
        chroma = np.array(rotate(QUALITIES[q], -k)).reshape(12, 1, 1)
        chroma = chroma - np.random.rand(12, 1, 1) / 2
        dataset.append([chroma, label])

    return pd.DataFrame(dataset, columns=["Chroma", "Label"])

COLORS = []
NUMBER_OF_NOTES = 24

for i in range(NUMBER_OF_NOTES):
    COLORS.append('#%06X' % random.randint(0, 0xFFFFFF))

def plot_cluster(title, filename, model, epoch, test_slices, batch_size, test_labels):
  z_test = model.encoder.predict(test_slices, batch_size=128)
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(z_test[2])

  x,y = principalComponents[:,0], principalComponents[:,1]
  x,y = z_test[2][:,0], z_test[2][:,1]
  labels = test_labels

  label_coords = {}

  for i in range(len(x)):
    if labels[i] in label_coords:
      label_coords[labels[i]].append((x[i],y[i]))
    else:
      label_coords[labels[i]] = [(x[i],y[i])]

  plt.figure(figsize=(12, 8), dpi=80)

  col = 0
  for l in label_coords.keys():
    centre = centroid(label_coords[l])
    coords = list(zip(*label_coords[l]))
    x = coords[0]
    y = coords[1]
    plt.scatter(x, y, c=COLORS[col], label=l,
                alpha=0.3, edgecolors='none')
    col+=1
    plt.annotate(l, (centre[0],centre[1]))

  plt.legend(loc="upper right")
  plt.xlim([-3, 3])
  plt.ylim([-3, 3])
  plt.savefig(f"./figures/{filename+str(epoch)}.png")
  plt.close()

def truncate(f, n):
    """
    Truncate/pad a float to a given number of decimal places without rounding.

    Args:
    f (float): The input float.
    n (int): The number of decimal places.

    Returns:
    str: The truncated/padded float as a string.
    """
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.

    Args:
    vor (scipy.spatial.Voronoi): The input Voronoi diagram.
    radius (float, optional): The distance to 'points at infinity'.

    Returns:
    list: Indices of vertices in each revised Voronoi region.
    numpy.ndarray: Coordinates for revised Voronoi vertices. Same as coordinates of input vertices, with 'points at infinity' appended to the end.
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

def load_and_enrich_dataset(size_dataset):
    # Generate a synthetic dataset used for evaluation
    synthetic_df = generate_dataset(size_dataset)
    
    # Initialize empty lists for test slices and labels
    test_slices = []
    test_labels = []

    # Load the training slices
    slices_train_raw = np.load("slices_train.npy")
    slices_train = []

    # Enrich the chroma of each training slice
    for c in slices_train_raw:
        c_new = enrich_chroma(c)
        slices_train.append(c_new)
    slices_train = np.array(slices_train)

    # Extract the test slices and labels from the synthetic dataset
    for i, row in synthetic_df.iterrows():
        test_slices.append(row["Chroma"])
        test_labels.append(row["Label"])

    test_slices = np.array(test_slices)
    
    return slices_train, test_slices, test_labels