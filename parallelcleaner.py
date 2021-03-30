#######################################################################################################
# AUTHOR  : aswin.pradeep@tarento.com
# AIM     : Standalone python function to clean parallel dataframe
# USAGE   : df = parallelcleanerfn(df, "hi")
#######################################################################################################

import pandas as pd
import urllib.request
import json
import numpy as np
import re
from datetime import datetime
import os
from bs4 import BeautifulSoup
from polyglot.detect import Detector
from tqdm import tqdm
import regxlist
import argparse

import warnings
warnings.filterwarnings("ignore")
# from langdetect import detect ,detect_langs ,DetectorFactory
# DetectorFactory.seed = 0

DATA_DIR = "train/all"

def parallelcleanerfn(df, secondlanguage):

    # df=pd.DataFrame()
    # df['L1']=dfx[dfx.columns[0]]
    # df['L2']=dfx[dfx.columns[1]]

    print("Progressing paralell cleanup script, No of rows:", len(df))

    df = df.replace('\n', '', regex=True)
    # drop duplicate pairs
    df = df.drop_duplicates(subset=['L2', 'L1'], keep="first")
    df_copy = df.copy()
    # create list of original content for dumping later
    dumpL1list = df_copy['L1'].to_list()
    dumpL2list = df_copy['L2'].to_list()

    # fixes html tags displayed in encoded format
    df = df.applymap(lambda text: BeautifulSoup(text, features="lxml").string)

    # replaces all non ascii characters from english column
    df["L1"] = df['L1'].str.replace('[^\x00-\x7F]', '')
    # replaces semi colon with full stop
    df["L1"] = df['L1'].str.replace(';', '.')
    df["L2"] = df['L2'].str.replace(';', '.')

    # calls list of items for replacement from file
    common_regList = regxlist.common_regList
    regList = regxlist.regList

    # replaces all cases with space
    for reg in common_regList:
        df['L2'] = df['L2'].str.replace(reg, ' ')
        df['L1'] = df['L1'].str.replace(reg, ' ')

    df['L2'] = df['L2'].str.strip()
    df['L1'] = df['L1'].str.strip()

    for reg in regList:
        df['L2'] = df['L2'].str.replace(reg, ' ')
        df['L1'] = df['L1'].str.replace(reg, ' ')

    df['L2'] = df['L2'].str.strip()
    df['L1'] = df['L1'].str.strip()
    L1list = df['L1'].to_list()
    L2list = df['L2'].to_list()

    newlanglist1 = []
    newlanglist2 = []
    dumplstc1 = []
    dumplstc2 = []

    # function identifies and drops non english content
    # erroneous sentences are dumped to a seperate csv file
    for i in tqdm(range(0, len(L1list))):

        try:

            title1 = L1list[i]
            title2 = L2list[i]
            dumptitle1 = dumpL1list[i]
            dumptitle2 = dumpL2list[i]

            if(len(title1) < 5 or len(re.findall(r'\w+', title1)) < 2 or len(title2) < 5):
                dumplstc1.append(dumptitle1)
                dumplstc2.append(dumptitle2)

            else:

                # using langdetect library. slower, more efficient. preferred for small datasets
                # detlan1=str(detect(title1))
                # detlan2=str(detect(title2))

                # using polygot library , faster , preferred for larger datasets.
                detlan1 = Detector(title1).language.code
                detlan2 = Detector(title2).language.code

                if(detlan1 != 'en' or detlan2 != secondlanguage):

                    dumplstc1.append(dumptitle1)
                    dumplstc2.append(dumptitle2)

                elif(detlan1 == 'en' and detlan2 == secondlanguage):

                    newlanglist1.append(title1)
                    newlanglist2.append(title2)

                else:
                    dumplstc1.append(dumptitle1)
                    dumplstc2.append(dumptitle2)

        except:

            dumplstc1.append(dumptitle1)
            dumplstc2.append(dumptitle2)

    cleaneddf = pd.DataFrame(
        list(zip(newlanglist1, newlanglist2)), columns=['L1', 'L2'])
    cleaneddf = cleaneddf.replace('\n', '', regex=True)

    # saves the dropped content to a dump CSV file
    dumpname = "Dumps/dump_" + \
        str(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")) + ".csv"
    dumpdf = pd.DataFrame(list(zip(dumplstc1, dumplstc2)),
                          columns=['L1', 'L2'])

    if not os.path.exists(os.path.dirname(dumpname)):
        try:
            os.makedirs(os.path.dirname(dumpname))
        except:
            print("Create a folder named  'Dumps' in script directory")

    dumpdf = dumpdf.replace('\n', '', regex=True)
    dumpdf.to_csv(dumpname, index=False)

    print("cleanup done, number of rows processed : ", len(cleaneddf))
    return(cleaneddf)

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


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # non_empty_lines = []
    # # for line in lines:
    # #     if line.strip() == '':
    # #         continue
    # #     non_empty_lines.append(line)
    # # return non_empty_lines
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Merge lang files and clean the corpus')
    parser.add_argument('--langs', type=str, default='all',
                        help='comma seperated string of languages you want to run merge for')

    args = parser.parse_args()
    langs = args.langs
    if langs == 'all':
        langs = "as, bn, gu, hi, kn, ml, mr, or, pa, ta, te"

    langs = [lang.strip() for lang in langs.split(',')]

    for lang in tqdm(langs):
        data_folder = f'{DATA_DIR}/en-{lang}'
        src_file = f'{data_folder}/train.en'
        tgt_file = f'{data_folder}/train.{lang}'
        clean_tsv_path = data_folder + f'/en-{lang}.tsv'

        filelist = [src_file, tgt_file]

        src_lines = [line.strip('\n') for line in read_lines(src_file)]
        tgt_lines = [line.strip('\n') for line in read_lines(tgt_file)]

        df = pd.DataFrame({'L1': src_lines, 'L2': tgt_lines}, columns=['L1', 'L2'])

        # df1 = pd.read_csv(filelist[0], sep='\n', names=['L1'])
        # df2 = pd.read_csv(filelist[1], sep='\n', names=['L2'])
        # df = pd.concat([df1, df2], axis=1)
        before_shape = df.shape
        df = df.dropna()
        print(lang, before_shape, df.shape)
        df_cleaned = parallelcleanerfn(df, lang)
        df_cleaned = df_cleaned.dropna()
        df_cleaned = df_cleaned.drop_duplicates()
        df_cleaned.to_csv(clean_tsv_path, sep='\t', header=None, index=False)
        separate_corpus(0, clean_tsv_path, src_file + '.clean')
        separate_corpus(1, clean_tsv_path, tgt_file + '.clean')
