import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture


def gaussian_likelihood(x, mean, variance):
    # Compute the probability density function (PDF)
    coef = 1.0 / np.sqrt(2.0 * np.pi * variance)
    exponent = np.exp(- (x - mean) ** 2 / (2 * variance))
    return coef * exponent

def pad_word(word):
    return np.pad(word,
                  np.round([s/10 for s in word.shape]).astype('int'),
                  'constant', constant_values=255)

def segmentWordsInSentence(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Split into rows
    row_sum = np.sum(255 - img, 1)
    n_pixels_allowed_in_zero_row = 1
    zero_rows = np.where(row_sum <= 255 * n_pixels_allowed_in_zero_row)[0]
    non_zero_segments = [[zero_rows[i], zero_rows[i + 1]] for i in range(len(zero_rows) - 1) if
                         zero_rows[i + 1] - zero_rows[i] > 1]
    rows = [img[s[0]:s[1], :] for s in non_zero_segments]

    # In each row, find the lengths and locations of spaces
    all_space_lengths = []
    all_space_start = []
    all_space_end = []
    for ri, row in enumerate(rows):
        col_sum = np.sum(255 - row, 0)
        col_sum[1:-1][(col_sum[:-2] == 0) & (col_sum[2:] == 0)] = 0 # if there's only one column with values, it's still space
        col_sum[2:-2][(col_sum[:-4] == 0) & (col_sum[4:] == 0)] = 0 # if there's only 2 columns with values, it's still space

        # Trim first and last space (their lengths are out of the distribution)
        start_ind = next((i for i, x in enumerate(col_sum) if x != 0), None)
        end_ind = len(col_sum) - 1 - next((i for i, x in enumerate(np.flip(col_sum)) if x != 0), None)
        col_sum = col_sum[start_ind:end_ind]
        rows[ri] = row[:, start_ind:end_ind]

        # Find start, end and lengths of spaces
        space_indicator = col_sum == 0
        space_start = np.where(space_indicator[1:] & ~space_indicator[:-1])[0] + 1
        space_end = np.where(~space_indicator[1:] & space_indicator[:-1])[0]
        space_lengths = space_end - space_start

        # Ignore random spaces
        space_start = space_start[space_lengths > 0]
        space_end = space_end[space_lengths > 0]
        space_lengths = space_lengths[space_lengths > 0]

        all_space_start.append(space_start)
        all_space_end.append(space_end)
        all_space_lengths.append(space_lengths)

    if len(rows) == 1 and len(all_space_lengths[0]) < 4:
        # Check if there's only one word
        # TODO: design a better test to this
        return [pad_word(rows[0])]

    # Assuming the spaces belong to 2 groups: spaces between words, and spaces between letters.
    # The spaces between words are larger than the spaces between letters.
    # To cluster them, we'll use KMeans.
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.concatenate(all_space_lengths).reshape(-1, 1))

    # Now we can get cropped images of the words.
    words = []
    for ri,row in enumerate(rows):
        real_space = kmeans.predict(all_space_lengths[ri].reshape(-1,1)) == np.argmax(kmeans.cluster_centers_)
        word_end = all_space_start[ri][real_space]
        word_end = np.append(word_end, row.shape[1])
        word_start = all_space_end[ri][real_space]
        word_start = np.insert(word_start, 0, 0)
        row_words = list(reversed([row[:,s:e] for s,e in zip(word_start, word_end)])) # reversing because Hebrew writes from right to left
        words = words + row_words

    words = [pad_word(w) for w in words]
    return words
