from music21 import corpus, pitch, interval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

SIZE_OF_DATASET = 300

class Piece:
    def __init__(self, title, composer, original_key=None, chromas=None, note_matrix=None):
        """
        Represents a musical piece.

        Args:
            title (str): The title of the piece.
            composer (str): The composer of the piece.
            original_key (str, optional): The original key of the piece. Defaults to None.
            chromas (ndarray, optional): The chromas of the piece. Defaults to None.
            note_matrix (ndarray, optional): The note matrix of the piece. Defaults to None.
        """
        self.title = title
        self.composer = composer
        self.note_matrix = note_matrix
        self.original_key = original_key
        self.chromas = chromas

    def compute_slices(self):
        """
        Computes the slices of the note matrix.
        """
        slices = []
        for k in range(0, self.note_matrix.shape[1] - 4, 4):
            s = self.note_matrix[:, k:k + 4]
            slices.append(s)
        self.slices = slices

    def compute_slices_old(self):
        """
        Obsolete.
        """
        slices = []
        sliced = []
        for j in range(0, self.note_matrix.shape[1]):
            chord = self.note_matrix[:, j]
            for i in chord:
                if i == 2:  # s'il y a un evenement "note_on"
                    if j not in sliced:
                        output_slice = (chord > 0) * np.ones_like(chord)
                        output_slice_image = np.zeros((128, 4, 1))
                        for k in range(4):
                            output_slice_image[:, k, :] = output_slice.reshape(128, 1)
                        slices.append(output_slice_image)
                    sliced.append(j)
        self.slices = slices

    def compute_chroma_slices(self):
        """
        Computes the chroma slices of the chromas.
        """
        slices = []
        for k in range(0, self.chromas.shape[1]):
            s = self.chromas[:, k].reshape(12, 1)
            slices.append(s)
        self.chroma_slices = slices

    def plot(self):
        """
        Plots the note matrix.
        """
        plt.imshow(self.note_matrix)
        plt.title(self.title + ", " + self.composer)
        plt.gca().invert_yaxis()
        plt.show()

def create_chorales_dataset(c_transposed=True, max_size=np.infty, all_transposed=False):
    """
    Creates a dataset of chorales.

    Args:
        c_transposed (bool, optional): Whether to transpose the chorales to C major or A minor. Defaults to True.
        max_size (int, optional): The maximum size of the dataset. Defaults to np.infty.
        all_transposed (bool, optional): Whether to include all transpositions of the chorales. Defaults to False.

    Returns:
        DataFrame: The dataset of chorales.
    """
    start_time = time.time()

    choral_dataset = []
    for choral in corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream'):
        if len(choral.parts) == 4:
            if len(choral_dataset) > max_size:
                break
            title = "BWV " + choral.metadata.title
            print(f"Adding choral {title} to dataset")
            composer = "J.S. Bach"
            key = choral.analyze('key')
            if all_transposed:
                for k in range(-6, 6):
                    choral_transposed = choral.transpose(k)
                    key = choral_transposed.analyze('key')
                    choral_matrix = np.zeros((128, int(4 * choral_transposed.duration.quarterLength)))
                    for voice in choral_transposed.parts:
                        beat_16 = 0
                        for n in voice.flat.notes:  # n is a note
                            n_midi = n.pitch.midi
                            n_length = 4 * n.duration.quarterLength  # length in 16th notes
                            for i in range(int(n_length)):
                                if i == 0:
                                    choral_matrix[n_midi][beat_16 + i] = 2
                                else:
                                    choral_matrix[n_midi][beat_16 + i] = 1
                            beat_16 += int(n_length)
                    c = Piece(title, composer, note_matrix=choral_matrix)
                    choral_dataset.append([c.title, c.composer, c])
            else:
                if c_transposed:
                    if key.mode == "major":
                        i = interval.Interval(key.tonic, pitch.Pitch('C'))
                    else:
                        i = interval.Interval(key.tonic, pitch.Pitch('A'))
                    choral = choral.transpose(i)
                choral_matrix = np.zeros((128, int(4 * choral.duration.quarterLength)))  # each column is a 16th note
                for voice in choral.parts:
                    beat_16 = 0
                    for n in voice.flat.notes:  # n is a note
                        n_midi = n.pitch.midi
                        n_length = 4 * n.duration.quarterLength  # length in 16th notes
                        for i in range(int(n_length)):
                            if i == 0:
                                choral_matrix[n_midi][beat_16 + i] = 2
                            else:
                                choral_matrix[n_midi][beat_16 + i] = 1
                        beat_16 += int(n_length)
                c = Piece(choral_matrix, title, composer, key)
                choral_dataset.append([c.title, c.original_key, c.length, c.composer, c])
    choral_df = pd.DataFrame(choral_dataset, columns=["Title", "Composer", "Piece object"])
    print("--- %s seconds ---" % (time.time() - start_time))
    return choral_df

def roll_to_chroma_squeeze(roll):
    """
    Reduces a piano roll modulo 12.

    Args:
        roll (ndarray): The piano roll.

    Returns:
        ndarray: The reduced piano roll.
    """
    res = np.zeros((12,1,1))
    for i in range(len(roll)):
        if roll[i][0][0] > 0:
            res[i%12][0][0] = roll[i][0][0]
    return res

if __name__ == '__main__':
    print("This process can take about 15 minutes")
    choral_df = create_chorales_dataset(max_size=SIZE_OF_DATASET, c_transposed=False, all_transposed=True)
    slices_train = []
    tones = []
    for c in choral_df["Piece object"]:
        c.compute_slices_old()
        slices_train += c.slices
    slices_train = np.array(slices_train).astype("float32") / np.amax(slices_train)  # map to [0,1], necessary in order to train the VAE
    slices_train = np.array([roll_to_chroma_squeeze(s) for s in slices_train])
    print("Data shape : "+str(slices_train.shape))
    np.save('slices_train', slices_train)



