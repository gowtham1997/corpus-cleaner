import os
from remove_train_devtest_overlaps import create_txt, read_lines
import random
import sys


if __name__ == "__main__":

    corpus_dir = sys.argv[1]

    for corpus in os.listdir(corpus_dir):
        data_dir = os.path.join(corpus_dir, corpus)
        lang_to_get_overlap = "en"
        source = corpus

        lang_pairs = os.listdir(data_dir)
        en_data = {}
        lang_data = {}
        req_testset_sizes = {}
        for pair in lang_pairs:
            if not os.path.exists(f"../devtest/{source}/{pair}"):
                os.makedirs(f"../devtest/{source}/{pair}")
            if not os.path.exists(f"../devtest/{source}/{pair}"):
                os.makedirs(f"../devtest/{source}/{pair}")
            lang = pair.split("-")[-1]
            en_data[pair] = read_lines(
                os.path.join(data_dir, pair, f"train.{lang_to_get_overlap}")
            )
            lang_data[pair] = read_lines(os.path.join(data_dir, pair, f"train.{lang}"))
            req_testset_sizes[pair] = int(0.05 * len(en_data[pair]))
            print(
                f"lang_pair: {pair}, Len of dataset: {len(en_data[pair])}, Required test size: {req_testset_sizes[pair]}"
            )

        train_pairs = {}
        dev_pairs = {}
        test_pairs = {}

        for pair in lang_pairs:
            train_pairs[pair] = []
            dev_pairs[pair] = []
            test_pairs[pair] = []

        if len(lang_pairs) > 1:
            # common english sentences between all en-lang pairs
            en_common = list(
                set.intersection(*[set(en_data[pair]) for pair in lang_pairs])
            )
            print("Number of en sentence overlaps between lang_pairs: ", len(en_common))
            dev_en_common, test_en_common = (
                en_common[: len(en_common) // 2],
                en_common[len(en_common) // 2 :],
            )
        else:
            dev_en_common = []
            test_en_common = []

        for pair in lang_pairs:
            for en_line, lang_line in zip(en_data[pair], lang_data[pair]):

                if en_line in dev_en_common:
                    # dont add to dev, if its already full
                    if len(dev_pairs[pair]) == req_testset_sizes[pair]:
                        train_pairs[pair].append((en_line, lang_line))
                        continue
                    dev_pairs[pair].append((en_line, lang_line))
                elif en_line in test_en_common:
                    if len(test_pairs[pair]) == req_testset_sizes[pair]:
                        train_pairs[pair].append((en_line, lang_line))
                        continue
                    test_pairs[pair].append((en_line, lang_line))
                else:
                    train_pairs[pair].append((en_line, lang_line))

        for pair in lang_pairs:
            train_pairs[pair] = list(set(train_pairs[pair]))
            dev_pairs[pair] = list(set(dev_pairs[pair]))
            test_pairs[pair] = list(set(test_pairs[pair]))

        # shuffle the train_set
        random.shuffle(train_pairs[pair])

        for pair in lang_pairs:
            req_size = int(0.1 * len(en_data[pair]))
            dev_req_size = int(req_size) // 2 - len(dev_pairs[pair])
            test_req_size = int(req_size) // 2 - len(test_pairs[pair])
            total_req_size = dev_req_size + test_req_size

            train_pairs[pair], devtest = (
                train_pairs[pair][total_req_size:],
                train_pairs[pair][:total_req_size],
            )
            # print(req_size, len(devtest))
            dev_additions, test_additions = (
                devtest[:dev_req_size],
                devtest[dev_req_size:],
            )
            dev_pairs[pair].extend(dev_additions)
            test_pairs[pair].extend(test_additions)

        for pair in lang_pairs:
            lang = pair.split("-")[-1]
            train_en, train_lang = zip(*train_pairs[pair])
            dev_en, dev_lang = zip(*dev_pairs[pair])
            test_en, test_lang = zip(*test_pairs[pair])

            print(lang, len(train_en), len(dev_en), len(test_en))
            create_txt(f"{data_dir}/{pair}/train.en", train_en)
            create_txt(f"{data_dir}/{pair}/train.{lang}", train_lang)

            create_txt(f"../devtest/{source}/{pair}/dev.en", dev_en)
            create_txt(f"../devtest/{source}/{pair}/dev.{lang}", dev_lang)

            create_txt(f"../devtest/{source}/{pair}/test.en", test_en)
            create_txt(f"../devtest/{source}/{pair}/test.{lang}", test_lang)
