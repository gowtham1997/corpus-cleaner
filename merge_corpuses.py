import os
import shutil
from tqdm import tqdm
import sys


LANGUAGES = ['en', 'as', 'bn', 'gu', 'hi',
             'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']


def read_lines(path):
    # if path doesnt exist, return empty list
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def create_txt(outFile, lines):
    add_newline = not '\n' in lines[0]
    outfile = open("{0}".format(outFile), "w")
    for line in lines:
        if add_newline:
            outfile.write(line + '\n')
        else:
            outfile.write(line)

    outfile.close()


def dedup(lang1_file, lang2_file):
    lang1_lines = read_lines(lang1_file)
    lang2_lines = read_lines(lang2_file)
    len_before = len(lang1_lines)

    deduped_lines = list(set(zip(lang1_lines, lang2_lines)))
    lang1_dedupped, lang2_dedupped = zip(*deduped_lines)

    len_after = len(lang1_dedupped)
    num_duplicates = len_before - len_after

    print(
        f'Dropped duplicate pairs in {lang1_file} Num duplicates -> {num_duplicates}')
    create_txt(lang1_file, lang1_dedupped)
    create_txt(lang2_file, lang2_dedupped)


def concatenate_txt_files(outfile, file_list):
    # given a list of txt files, merge them to outfile
    with open(outfile, "w") as wfd:
        for f in file_list:
            with open(f, "r") as fd:
                shutil.copyfileobj(fd, wfd)

# to merge all the training data of wat2021 and remove duplicates
def merge_corpus_dir(data_dir, split="train"):

    datasets = os.listdir(data_dir)

    # we will merge all the corpuses into a all folder.
    if 'all' in datasets:
        datasets.remove('all')
    for lang1 in tqdm(LANGUAGES):
        for lang2 in tqdm(LANGUAGES):
            if lang1 == lang2:
                continue
            lang1_files = []
            lang2_files = []
            output_dir = os.path.join(data_dir, f"all/{lang1}-{lang2}")
            for dataset in datasets:
                file1 = os.path.join(
                    data_dir, dataset, f"{lang1}-{lang2}", f"{split}.{lang1}"
                )
                file2 = os.path.join(
                    data_dir, dataset, f"{lang1}-{lang2}", f"{split}.{lang2}"
                )
                # to handle wat2021 dev or test sets
                file3 = os.path.join(
                    data_dir, dataset, f"{split}.{lang1}"
                )
                file4 = os.path.join(
                    data_dir, dataset, f"{split}.{lang2}"
                )
                if os.path.exists(file1) and os.path.join(file2):
                    lang1_files.append(file1)
                    lang2_files.append(file2)
                elif os.path.exists(file3) and os.path.join(file4):
                    lang1_files.append(file3)
                    lang2_files.append(file4)
            if lang1_files == [] or lang2_files == []:
                continue
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            outfile1 = os.path.join(output_dir, f"{split}.{lang1}")
            outfile2 = os.path.join(output_dir, f"{split}.{lang2}")
            concatenate_txt_files(outfile1, lang1_files)
            concatenate_txt_files(outfile2, lang2_files)
            dedup(outfile1, outfile2)


if __name__ == "__main__":
    corpus_dir = sys.argv[1]
    split = sys.argv[2]
    # merge_training_data('wat2021/train', 'train')
    merge_corpus_dir(corpus_dir, split)
