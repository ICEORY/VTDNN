import os

__all__ = ['loadLabels']


def loadLabels(file_path, position=1, split=" "):
    label = []
    if os.path.isfile(file_path):
        with open(file_path, "r") as fr:
            read_str = fr.readline()
            while read_str:
                read_str = read_str.split(split)
                label.append(read_str[position].replace("\n", ""))
                read_str = fr.readline()

    return label
