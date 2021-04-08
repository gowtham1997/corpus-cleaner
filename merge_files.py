import shutil
import pandas as pd
import os
# import tensorflow as tf
import re
from tqdm import tqdm
import argparse


# def _parse_sgm(path):
#     """Returns sentences from a single SGML file."""
#     lang = path.split(".")[-2]
#     sentences = []
#     # Note: We can't use the XML parser since some of the files are badly
#     # formatted.
#     seg_re = re.compile(r"<seg id=\"\d+\">(.*)</seg>")
#     with tf.io.gfile.GFile(path) as f:
#         for line in f:
#             seg_match = re.match(seg_re, line)
#             if seg_match:
#                 assert len(seg_match.groups()) == 1
#                 sentences.append(seg_match.groups()[0])
#     return sentences, lang


def count_lines(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


def concatenate_txt_files(outfile, file_list):
    # given a list of txt files, merge them to outfile
    with open(outfile, "w") as wfd:
        for f in file_list:
            with open(f, "r") as fd:
                shutil.copyfileobj(fd, wfd)


def remove_file(path):
    if not os.path.exists(path):
        os.remove(path)


# below is for dropping duplicate from text file
def drop_duplicate(inFile, outFile, return_lines=False):
    # taken from https://github.com/project-anuvaad/OpenNMT-py/blob/master/corpus/file_cleaner.py
    lines = set()
    lines_seen = set()  # holds lines already seen
    outfile = open("{0}".format(outFile), "w")
    for line in open("{0}".format(inFile), "r"):
        lines.add(line)
        if line not in lines_seen:  # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    print(
        f"Numbers of lines before and after removing duplicates in {inFile} is {count_lines(inFile)} --> {count_lines(outFile)}"
    )
    if return_lines:
        return lines
    return set()


def tab_separated_parllel_corpus(mono_corpus1, mono_corpus2, out_file):
    # taken from https://github.com/project-anuvaad/OpenNMT-py/blob/master/corpus/file_cleaner.py
    with open("{0}".format(mono_corpus1), 'r') as xh:
        with open("{0}".format(mono_corpus2), 'r') as yh:
            with open("{0}".format(out_file), "w") as zh:
                # Read first file
                xlines = xh.readlines()
                # Read second file
                ylines = yh.readlines()

                assert len(xlines) == len(
                    ylines), f"{len(xlines) != len(ylines)}"

                # Write to third file
                for i in range(len(xlines)):
                    line = ylines[i].strip() + "\t" + xlines[i]
                    zh.write(line)


# separation into master corpus src and tgt for training. After this tokenization needs to be done(indic nlp, moses),then feed into OpenNMT
def separate_corpus(col_num, inFile, outFile):
    # taken from https://github.com/project-anuvaad/OpenNMT-py/blob/master/corpus/file_cleaner.py
    outfile = open("{0}".format(outFile), "w")
    delimiter = "\t"
    for line in open("{0}".format(inFile), "r"):
        # col_data.append(f.readline().split(delimiter)[col_num])
        outfile.write(str(line.split(delimiter)[col_num].replace("\n", "")))
        outfile.write("\n")
    outfile.close()


def concatenate(data_dir, src_lang="en", tr_lang="hi",
                split="train", verbose=True):
    src_lang = "en"
    _dir = os.path.join(data_dir, f"all/{src_lang}-{tr_lang}")
    src_lang_files = []
    tr_lang_files = []

    # name of all corpuses
    tasks = os.listdir(data_dir)

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    # we will concatenate all txt files into all/{src_lang)-{tr_lang} folder
    if "all" in tasks:
        tasks.remove("all")
    tasks.sort()

    # gather all the txt files we need to merge
    for task in tasks:
        file1 = os.path.join(
            data_dir, task, f"{src_lang}-{tr_lang}", f"{split}.{src_lang}"
        )
        file2 = os.path.join(
            data_dir, task, f"{src_lang}-{tr_lang}", f"{split}.{tr_lang}"
        )
        if os.path.exists(file1) and os.path.join(file2):
            src_lang_files.append(file1)
            tr_lang_files.append(file2)
        else:
            # in some cases(WAT 2020 dev test) we dont have en-{lang} folder
            file1 = os.path.join(data_dir, task, f"{split}.{src_lang}")
            file2 = os.path.join(data_dir, task, f"{split}.{tr_lang}")
            if os.path.exists(file1) and os.path.exists(file2):
                src_lang_files.append(file1)
                tr_lang_files.append(file2)

    outfile1 = os.path.join(_dir, f"{split}.{src_lang}")
    outfile2 = os.path.join(_dir, f"{split}.{tr_lang}")
    concatenate_txt_files(outfile1, src_lang_files)
    concatenate_txt_files(outfile2, tr_lang_files)

    if verbose:
        # print stats of each corpus
        df1 = pd.DataFrame(src_lang_files + [outfile1], columns=["Filename"])
        df1["src_lines"] = df1["Filename"].map(count_lines)
        df2 = pd.DataFrame(tr_lang_files + [outfile2], columns=["Filename"])
        df2["tgt_lines"] = df2["Filename"].map(count_lines)
        df1["tgt_lines"] = df2["tgt_lines"]

        df1["Filename"] = df1["Filename"].map(
            lambda x: x.replace("\\", "/").split("/")[1])
        print(df1)
    tsv_path1 = os.path.join(
        _dir,
        f"{split}-{src_lang}-{tr_lang}.tsv",
    )
    tsv_path2 = os.path.join(
        _dir,
        f"{split}-{src_lang}-{tr_lang}-unique.tsv",
    )
    tab_separated_parllel_corpus(outfile1, outfile2, tsv_path1)
    assert count_lines(tsv_path1) > 0
    _ = drop_duplicate(tsv_path1, tsv_path2, return_lines=False)
    separate_corpus(1, tsv_path2, outfile1)
    separate_corpus(0, tsv_path2, outfile2)
    return df1
    # remove_file(tsv_path1)
    # remove_file(tsv_path2)


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


# def get_overlap(df1, df2):
#     return list(set(df1) & set(df2))

def get_overlap(df1, df2):
    len_before = len(df1)
    df1 = df1[~df1[0].isin(df2[0]) & ~df1[1].isin(df2[1])]
    len_after = len(df1)
    print(
        f'Sizes after overlap removal: {len_before} -> {len_after}. Number of overlaps: {len_before - len_after}')
    return df


def get_union(df1, df2):
    return list(set(df1) | set(df2))


# def remove_duplicates(df1, df2):
#     len_before = len(df1)
#     df = set(df1) - set(df2)
#     len_after = len(df)
#     print(
#         f'Sizes after overlap removal: {len_before} -> {len_after}. Number of overlaps: {len_before - len_after}')
#     return df

def remove_duplicates(df1, df2):
    # remove duplicates between df1 and df2 from df1
    len_before = len(df1)
    df1 = df1[~df1[0].isin(df2[0]) & ~df1[1].isin(df2[1])]
    len_after = len(df1)
    print(
        f'Sizes after overlap removal: {len_before} -> {len_after}. Number of overlaps: {len_before - len_after}')
    return df


def create_txt(outFile, lines, add_newline=False):
    outfile = open("{0}".format(outFile), "w")
    for line in lines:
        if add_newline:
            outfile.write(line + '\n')
        else:
            outfile.write(line)

    outfile.close()


def create_sep_corpuses(tsvpath, lines, outfile1, outfile2):
    # creates the tsv file containing parallel data and also seperate txt files
    # for each language

    create_txt(tsvpath, lines)
    # outfile1 is {split}.en
    separate_corpus(1, tsvpath, outfile1)
    # outfile1 is {split}.{lang}
    separate_corpus(0, tsvpath, outfile2)


def clean_and_sep_corpuses(langs):
    # removes devtest overlaps with train

    if isinstance(langs, str):
        langs = langs.split(',')
    for lang in tqdm(langs):
        test_outfile1 = f'devtest/all/en-{lang}/test.en'
        test_outfile2 = f'devtest/all/en-{lang}/test.{lang}'
        dev_outfile1 = f'devtest/all/en-{lang}/dev.en'
        dev_outfile2 = f'devtest/all/en-{lang}/dev.{lang}'
        train_outfile1 = f'train/all/en-{lang}/train.en'
        train_outfile2 = f'train/all/en-{lang}/train.{lang}'

        # dev = read_lines(f'devtest/all/en-{lang}/dev-en-{lang}-unique.tsv')
        # test = read_lines(f'devtest/all/en-{lang}/test-en-{lang}-unique.tsv')
        # train = read_lines(f'train/all/en-{lang}/train-en-{lang}-unique.tsv')
        dev = read_tsv(f'devtest/all/en-{lang}/dev-en-{lang}-unique.tsv')
        test = read_tsv(f'devtest/all/en-{lang}/test-en-{lang}-unique.tsv')
        train = read_tsv(f'train/all/en-{lang}/train-en-{lang}-unique.tsv')
        # remove overlap between dev and test
        dev = remove_duplicates(dev, test)
        create_sep_corpuses(
            f'devtest/all/en-{lang}/test-en-{lang}-final.tsv',
            test, test_outfile1, test_outfile2)
        create_sep_corpuses(
            f'devtest/all/en-{lang}/dev-en-{lang}-final.tsv',
            dev, dev_outfile1, dev_outfile2)

        devtest = get_union(dev, test)
        train = remove_duplicates(train, devtest)
        create_sep_corpuses(
            f'train/all/en-{lang}/en-{lang}-final.tsv',
            train, train_outfile1, train_outfile2)
        print(f"Final Counts for language {lang}:", len(
            train), len(dev), len(test))
        return len(train), len(dev), len(test)


def validate_corpus(lang1, lang2, train_corpus1, train_corpus2, dev_corpus1,
                    dev_corpus2, test_corpus1, test_corpus2):
    tmp_dir = 'temp_check_dir/'
    final_dir = tmp_dir + f'{lang1}-{lang2}/'
    os.makedirs(tmp_dir)
    os.makedirs(final_dir)

    def create_duplicate_free_tsv(corpus1, corpus2, split='train'):
        tab_separated_parllel_corpus(
            corpus1, corpus2, tmp_dir + f'{split}_merge1.tsv')
        _ = drop_duplicate(tmp_dir + f'{split}_merge1.tsv',
                           tmp_dir + f'{split}_merge2.tsv', return_lines=False)
        return tmp_dir + f'{split}_merge2.tsv'

    train_file = create_duplicate_free_tsv(
        train_corpus1, train_corpus2, split='train')
    dev_file = create_duplicate_free_tsv(dev_corpus1, dev_corpus2, split='dev')
    test_file = create_duplicate_free_tsv(
        test_corpus1, test_corpus2, split='test')

    train = read_lines(train_file)
    dev = read_lines(dev_file)
    test = read_lines(test_file)

    print('Removing overlaps between dev and test from dev:')
    dev = remove_duplicates(dev, test)

    devtest = get_union(dev, test)
    print('Removing overlaps between dev+test from train:')
    train = remove_duplicates(train, devtest)

    create_sep_corpuses(final_dir + f'{lang1}-{lang2}.tsv', train,
                        final_dir + f'train.{lang1}', final_dir + f'train.{lang2}')


def get_avg_word_count(lines):
    count = 0
    corpus_len = len(lines)
    for line in lines:
        count += len(line.split())
    return count / corpus_len


def get_avg_word_counts_for_corpuses(data_dir, src_lang="en", tr_lang="hi",
                                     split="train", verbose=True):
    src_lang = "en"
    _dir = os.path.join(data_dir, f"all/{src_lang}-{tr_lang}")
    src_lang_files = []
    tr_lang_files = []

    # name of all corpuses
    tasks = os.listdir(data_dir)
    tasks.sort()

    counts = {}

    # gather all the txt files we need to merge
    for task in tasks:
        if task.lower() == '.ds_store':
            continue
        file1 = os.path.join(
            data_dir, task, f"{src_lang}-{tr_lang}", f"{split}.{src_lang}"
        )
        file2 = os.path.join(
            data_dir, task, f"{src_lang}-{tr_lang}", f"{split}.{tr_lang}"
        )
        if os.path.exists(file1) and os.path.join(file2):
            src_lang_files.append(file1)
            tr_lang_files.append(file2)
        else:
            # in some cases(WAT 2020 dev test) we dont have en-{lang} folder
            file1 = os.path.join(data_dir, task, f"{split}.{src_lang}")
            file2 = os.path.join(data_dir, task, f"{split}.{tr_lang}")
            if not (os.path.exists(file1) and os.path.join(file2)):
                continue
        contents1 = read_lines(file1)
        contents2 = read_lines(file2)
        len_content = len(contents1)
        count_src = get_avg_word_count(contents1)
        count_tr = get_avg_word_count(contents2)
        counts[task + f' ({len_content})'] = {f'{src_lang} avg sentence len': count_src,
                                              f'{tr_lang} avg sentence len': count_tr}
        df = pd.DataFrame(counts).transpose()
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='merge txt files from multiple folders and remove duplicates')
    parser.add_argument('--langs', type=str, default='all',
                        help='comma seperated string of languages you want to run merge for')

    args = parser.parse_args()
    langs = args.langs
    if langs == 'all':
        langs = "as, bn, gu, hi, kn, ml, mr, or, pa, ta, te"

    langs = [lang.strip() for lang in langs.split(',')]

    en_langs = ['en-' + lang for lang in langs]

    train_datasets = os.listdir('train')
    devtest_datasets = os.listdir('devtest')
    # to preserver ordering while printing
    if 'all' in train_datasets:
        train_datasets.remove('all')
    if 'all' in devtest_datasets:
        devtest_datasets.remove('all')
    train_datasets.sort()
    devtest_datasets.sort()
    train_datasets.append('all')
    devtest_datasets.append('all')

    final_train_dataset_stats = pd.DataFrame(
        index=train_datasets, columns=en_langs).fillna(0)
    final_dev_dataset_stats = pd.DataFrame(
        index=devtest_datasets, columns=en_langs).fillna(0)
    final_test_dataset_stats = pd.DataFrame(
        index=devtest_datasets, columns=en_langs).fillna(0)
    # this just merges all corpuses into an all folder and removes duplicates
    for lang in langs:
        for split in ['train', 'dev', 'test']:
            if split == 'train':
                data_dir = 'train'
            else:
                data_dir = 'devtest'
            print(f'Concatenating {data_dir} folder of lang {lang}')
            stats = concatenate(data_dir=data_dir, src_lang='en',
                                tr_lang=lang, split=split)

            datasets = list(stats['Filename'])
            stats.set_index('Filename', inplace=True)
            if split == 'train':
                for task in datasets:
                    if task == 'all':
                        continue
                    final_train_dataset_stats.loc[task, 'en-' +
                                                  lang] = stats.loc[task, 'tgt_lines']
            elif split == 'dev':
                for task in datasets:
                    if task == 'all':
                        continue
                    final_dev_dataset_stats.loc[task, 'en-' +
                                                lang] = stats.loc[task, 'tgt_lines']
            else:
                for task in datasets:
                    if task == 'all':
                        continue
                    final_test_dataset_stats.loc[task, 'en-' +
                                                 lang] = stats.loc[task, 'tgt_lines']
        # uses the concatenated txt files in all folder to remove devtest overlaps
        all_train, all_dev, all_test = clean_and_sep_corpuses(lang)
        final_train_dataset_stats.loc['all', 'en-' + lang] = all_train
        final_dev_dataset_stats.loc['all', 'en-' + lang] = all_dev
        final_test_dataset_stats.loc['all', 'en-' + lang] = all_test

    print(final_train_dataset_stats)
    print()
    print(final_dev_dataset_stats)
    print()
    print(final_test_dataset_stats)

    final_train_dataset_stats.to_csv('train_stats.csv')
    final_dev_dataset_stats.to_csv('dev_stats.csv')
    final_test_dataset_stats.to_csv('test_stats.csv')
