from glob import glob
import os

import numpy as np


if __name__ == "__main__":
    folders = glob('out/*')

    chunks_folder = 'chunks'

    os.makedirs(chunks_folder, exist_ok=True)

    n_chunks = 10

    step = int(np.ceil(len(folders) / n_chunks))

    for i in range(0, len(folders), step):
        filename = f"{i}-{i+step}-chunk.txt"
        with open(os.path.join(chunks_folder, filename), "w") as out_file:
            out_file.write('\n'.join(folders[i:i+step]))