from collections import OrderedDict
import re
import numpy as np
from scipy import optimize
import pickle
import scipy.special as special
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd
import time
import cProfile


# TODO remove unnecessary imports
# TODO delete print
# TODO delete time inside functions
# TODO change .pkl weights name to: trained_weights_data_i.pkl

class ClassStatistics:
    """
    define classes of features and its statistics (e.g. counts)
    """

    def __init__(self, file_path):
        """
        :param file_path: full path of the train file to read
        """
        self.file_path = file_path
        self.Y = set()  # all the different tags we saw in training

        # Init all features dictionaries
        self.class100_dict = OrderedDict()  # {(100, word, tag): # times seen}
        self.class101_dict = OrderedDict()  # {(101, word[- length:], tag): # times seen}
        self.class102_dict = OrderedDict()  # {(102, word[:length], tag): # times seen}
        self.class103_dict = OrderedDict()  # {(103, tag-2, tag-1, tag): # times seen}
        self.class104_dict = OrderedDict()  # {(104, tag-1, tag): # times seen}
        self.class105_dict = OrderedDict()  # {(105, tag): # times seen}

        # Additional
        self.class106_dict = OrderedDict()  # {(106, word-1, tag): # times seen}
        self.class107_dict = OrderedDict()  # {(107, word+1, tag): # times seen}

        # Numbers
        self.class108_dict = OrderedDict()  # {(108.x, tag): # times seen}

        # Capital letters
        self.class109_dict = OrderedDict()  # {(109.x, prev_tag, cur_tag): # times seen}
        self.class110_dict = OrderedDict()  # {(110.x, tag): # times seen}  # TODO Question for Gal: is it correct to be under #Capital?

    def set_class100_dict(self):
        """
            Create counts dict for class 100 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    self.Y.add(cur_tag)
                    if (100, cur_word, cur_tag) not in self.class100_dict:
                        self.class100_dict[(100, cur_word, cur_tag)] = 1
                    else:
                        self.class100_dict[(100, cur_word, cur_tag)] += 1
        self.Y = sorted(list(self.Y))

    def set_class101_dict(self):
        """
            Create counts dict for class 101 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if not re.match('^[0-9]+([-,.:]?[0-9]?)*$', cur_word) and not re.match('^[0-9]+\\\\/[0-9]+$',
                                                                                           cur_word):
                        n = min(len(cur_word) - 1, 7)
                        for suffix_length in range(1, n + 1):
                            if not re.match('^[0-9]+$', cur_word[-suffix_length:]):
                                if (101, cur_word[-suffix_length:], cur_tag) not in self.class101_dict:
                                    self.class101_dict[(101, cur_word[-suffix_length:], cur_tag)] = 1
                                else:
                                    self.class101_dict[(101, cur_word[-suffix_length:], cur_tag)] += 1

    def set_class102_dict(self):
        """
            Create counts dict for class 102 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if not re.match('^[0-9]+([-,.:]*[0-9]*)*$', cur_word) and not re.match('^[0-9]+\\\\/[0-9]+$',
                                                                                           cur_word):
                        n = min(len(cur_word) - 1, 7)
                        for prefix_length in range(1, n + 1):
                            if not re.match('^[0-9]+$', cur_word[:prefix_length]):
                                if (102, cur_word[:prefix_length], cur_tag) not in self.class102_dict:
                                    self.class102_dict[(102, cur_word[:prefix_length], cur_tag)] = 1
                                else:
                                    self.class102_dict[(102, cur_word[:prefix_length], cur_tag)] += 1

    def set_class103_dict(self):
        """
            Create counts dict for class 103 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    prev_tag = splited_words[word_idx - 1].split('_')[1] if word_idx > 0 else ''
                    prev_prev_tag = splited_words[word_idx - 2].split('_')[1] if word_idx - 1 > 0 else ''
                    if (103, prev_prev_tag, prev_tag, cur_tag) not in self.class103_dict:
                        self.class103_dict[(103, prev_prev_tag, prev_tag, cur_tag)] = 1
                    else:
                        self.class103_dict[(103, prev_prev_tag, prev_tag, cur_tag)] += 1

    def set_class104_dict(self):
        """
            Create counts dict for class 104 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    prev_tag = splited_words[word_idx - 1].split('_')[1] if word_idx > 0 else ''
                    if (104, prev_tag, cur_tag) not in self.class104_dict:
                        self.class104_dict[(104, prev_tag, cur_tag)] = 1
                    else:
                        self.class104_dict[(104, prev_tag, cur_tag)] += 1

    def set_class105_dict(self):
        """
            Create counts dict for class 105 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    if (105, cur_tag) not in self.class105_dict:
                        self.class105_dict[(105, cur_tag)] = 1
                    else:
                        self.class105_dict[(105, cur_tag)] += 1

    def set_class106_dict(self):
        """
            Create counts dict for class 106 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    prev_word = splited_words[word_idx - 1].split('_')[0] if word_idx != 0 else '*'
                    if (106, prev_word, cur_tag) not in self.class106_dict:
                        self.class106_dict[(106, prev_word, cur_tag)] = 1
                    else:
                        self.class106_dict[(106, prev_word, cur_tag)] += 1

    def set_class107_dict(self):
        """
            Create counts dict for class 107 features
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    next_word = splited_words[word_idx + 1].split('_')[0] if \
                        word_idx != len(splited_words) - 1 else 'STOP'
                    if (107, next_word, cur_tag) not in self.class107_dict:
                        self.class107_dict[(107, next_word, cur_tag)] = 1
                    else:
                        self.class107_dict[(107, next_word, cur_tag)] += 1

    def set_class108_dict(self):
        """
            Create counts dict for class 108 features - Numbers
            define number = [numbers][\/]?[numbers]
            1. if word is only number or [-.:,\/%] chars
            2. elif word is [letters][-.][number]([-.]*[letters]+)+
                by division to the final three chars is letters or not
            3. elif word is [letters][-.][number]
                by division based on first char
            4. elif word is [number][-.][letters]([-.]*[number]+)+
            5. elif word is [number][-.][letters]
            6. if word is year pattern  --> e.g. 1980s , mid-1980, ‘80s
            7. any other pattern with digits
                by division to the final three chars is letters or not
                and amount of '-' chars
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    cur_word = splited_words[word_idx].split('_')[0]
                    hyphen_count = cur_word.count('-')
                    if re.search(r'\d', cur_word):
                        splited_cur_word = re.split('-', cur_word)
                        no_category = True
                        if re_match_words('^[0-9]+$', re.split('[-]|[,]|[.]|[:]|[\\\\]|[/]|[%]', cur_word)):
                            if (108.1, cur_tag) not in self.class108_dict:
                                self.class108_dict[(108.1, cur_tag)] = 1
                            else:
                                self.class108_dict[(108.1, cur_tag)] += 1
                            no_category = False

                        elif hyphen_count > 1 and re_match_letters_numbers(['^[A-Za-z]+$', '^[0-9]+$'],
                                                                           splited_cur_word) \
                                and not cur_word.startswith('mid'):
                            if re.match('^[A-Za-z]+$', splited_cur_word[-1]):
                                if (108.21, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.21, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.21, cur_tag)] += 1
                            else:
                                if (108.22, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.22, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.22, cur_tag)] += 1
                            no_category = False

                        elif hyphen_count == 1 and re_match_letters_numbers(['^[A-Za-z]+$', '^[0-9]+$'],
                                                                            splited_cur_word):
                            if re.match('^[a-z]$', cur_word[0]):
                                if (108.31, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.31, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.31, cur_tag)] += 1
                            else:
                                if (108.32, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.32, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.32, cur_tag)] += 1
                            no_category = False

                        elif hyphen_count > 1 and re_match_numbers_letters(['^[0-9]+$', '^[A-Za-z]+$'],
                                                                           splited_cur_word):
                            if (108.4, cur_tag) not in self.class108_dict:
                                self.class108_dict[(108.4, cur_tag)] = 1
                            else:
                                self.class108_dict[(108.4, cur_tag)] += 1
                            no_category = False

                        elif hyphen_count == 1 and re_match_numbers_letters(['^[0-9]+$', '^[A-Za-z]+$'],
                                                                            splited_cur_word):
                            if (108.5, cur_tag) not in self.class108_dict:
                                self.class108_dict[(108.5, cur_tag)] = 1
                            else:
                                self.class108_dict[(108.5, cur_tag)] += 1
                            no_category = False

                        if re.match('^([a-zA-Z]*(-)?[0-9][0-9][0-9][0-9](s?))$|^(\'[0-9][0-9]s)$', cur_word):
                            if (108.6, cur_tag) not in self.class108_dict:
                                self.class108_dict[(108.6, cur_tag)] = 1
                            else:
                                self.class108_dict[(108.6, cur_tag)] += 1
                            no_category = False

                        if no_category:
                            if re.match('^[A-Za-z]+$', splited_cur_word[-1]):
                                if (108.7, hyphen_count, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.7, hyphen_count, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.7, hyphen_count, cur_tag)] += 1
                            else:
                                if (108.8, hyphen_count, cur_tag) not in self.class108_dict:
                                    self.class108_dict[(108.8, hyphen_count, cur_tag)] = 1
                                else:
                                    self.class108_dict[(108.8, hyphen_count, cur_tag)] += 1

    def set_class109_dict(self):
        """
        We divide to different sub-classes the treatment for capital letters
            1. if word is not the first, starts with capital and has [- .]+ and both next and previous words starts with capital
            2. elif word is not the first, starts with capital and has [- .]+ and prev starts with capital
            3. elif word is not the first, starts with capital and has [- .]+
            4. elif word is not the first, starts with capital and both next and previous words starts with capital
            5. elif word is the first, starts with capital and has [- .]+ and next and prev word starts with capital
            6. elif word is the first, starts with capital and has [- .]+ and next word starts with capital
            7. elif word is the first, starts with capital and has [- .]+
            8. elif word is the first, starts with capital and next and prev word starts with capital
            9. elif word is only capital and has more than one letter, next word also, and also prev word
            11. elif word is only capital and has more than one letter, next word also
            12. elif word is only capital and has more than one letter
            13. if just has some capital somewhere, and count number of '-' chars
            14. if word has capital after small letter
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    cur_word = splited_words[word_idx].split('_')[0]
                    prev_word = splited_words[word_idx - 1].split('_')[0] if word_idx > 0 else '*'
                    prev_tag = splited_words[word_idx - 1].split('_')[1] if word_idx > 0 else ''
                    next_word = splited_words[word_idx + 1].split('_')[0] if word_idx + 1 < len(
                        splited_words) - 1 else 'STOP'
                    hyphen_count = cur_word.count('-')
                    if word_idx > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word) \
                            and re.match('^[A-Z]', prev_word):
                        if (109.1, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.1, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.1, prev_tag, cur_tag)] += 1

                    elif word_idx > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', prev_word):
                        if (109.2, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.2, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.2, prev_tag, cur_tag)] += 1

                    elif word_idx > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word):
                        if (109.3, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.3, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.3, prev_tag, cur_tag)] += 1

                    elif word_idx > 0 and re.match('^[A-Z]', cur_word) and re.match('^[A-Z]', next_word) \
                            and re.match('^[A-Z]', prev_word):
                        if (109.4, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.4, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.4, prev_tag, cur_tag)] += 1

                    elif word_idx == 0 or (word_idx > 0 and prev_word in ['``', '.']):
                        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word) \
                                and re.match('^[A-Z]', prev_word):
                            if (109.5, prev_tag, cur_tag) not in self.class109_dict:
                                self.class109_dict[(109.5, prev_tag, cur_tag)] = 1
                            else:
                                self.class109_dict[(109.5, prev_tag, cur_tag)] += 1

                    elif word_idx == 0 or (word_idx > 0 and prev_word in ['``', '.']):
                        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word):
                            if (109.6, prev_tag, cur_tag) not in self.class109_dict:
                                self.class109_dict[(109.6, prev_tag, cur_tag)] = 1
                            else:
                                self.class109_dict[(109.6, prev_tag, cur_tag)] += 1

                    elif word_idx == 0 or (word_idx > 0 and prev_word in ['``', '.']):
                        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word):

                            if (109.7, prev_tag, cur_tag) not in self.class109_dict:
                                self.class109_dict[(109.7, prev_tag, cur_tag)] = 1
                            else:
                                self.class109_dict[(109.7, prev_tag, cur_tag)] += 1

                    elif word_idx == 0 or (word_idx > 0 and prev_word in ['``', '.']):
                        if re.match('^[A-Z]', cur_word) and re.match('^[A-Z]', next_word) \
                                and re.match('^[A-Z]', prev_word):
                            if (109.8, prev_tag, cur_tag) not in self.class109_dict:
                                self.class109_dict[(109.8, prev_tag, cur_tag)] = 1
                            else:
                                self.class109_dict[(109.8, prev_tag, cur_tag)] += 1

                    elif re.match('^[A-Z][A-Z]+$', cur_word) and re.match('^[A-Z][A-Z]+$', next_word) and re.match(
                            '^[A-Z][A-Z]+$', prev_word):
                        if (109.9, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.9, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.9, prev_tag, cur_tag)] += 1

                    elif re.match('^[A-Z][A-Z]+$', cur_word) and re.match('^[A-Z][A-Z]+$', next_word):

                        if (109.11, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.11, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.11, prev_tag, cur_tag)] += 1

                    elif re.match('^[A-Z][A-Z]+$', cur_word):

                        if (109.12, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.12, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.12, prev_tag, cur_tag)] += 1

                    if re.match('[A-Z]', cur_word):

                        if (109.13, hyphen_count, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.13, hyphen_count, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.13, hyphen_count, prev_tag, cur_tag)] += 1

                    if re.match('(.*?)[a-z](.*?)[A-Z]', cur_word):
                        if (109.14, prev_tag, cur_tag) not in self.class109_dict:
                            self.class109_dict[(109.14, prev_tag, cur_tag)] = 1
                        else:
                            self.class109_dict[(109.14, prev_tag, cur_tag)] += 1

    def set_class110_dict(self):
        """ TODO update documentation
        originally small model
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    cur_word = splited_words[word_idx].split('_')[0]
                    # 110.1 + 110.2
                    if len(cur_word) >= 13:
                        if re.search('ing$', cur_word):
                            if (110.11, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.11, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.11, cur_tag)] += 1

                        elif re.search('ed$', cur_word):
                            if (110.14, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.14, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.14, cur_tag)] += 1

                        elif re.search('ally$', cur_word) or re.search('ely$', cur_word) or \
                                re.search('tly$', cur_word):
                            if (110.12, cur_tag) not in self.class110_dict:  # RB tag
                                self.class110_dict[(110.12, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.12, cur_tag)] += 1

                        elif re.search('tant$', cur_word) or \
                                re.search('cal$', cur_word) or \
                                re.search('ic$', cur_word) or \
                                re.search('ive$', cur_word) or \
                                re.search('nal$', cur_word) or \
                                re.search('-dependent$', cur_word) or \
                                re.search('-sensitive$', cur_word) or \
                                re.search('-specific$', cur_word) or \
                                re.search('tly$', cur_word):  # JJ tag
                            if (110.13, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.13, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.13, cur_tag)] += 1
                        else:  # NN + NNS tags
                            if cur_word[-1] == 's':
                                if (110.2, cur_tag) not in self.class110_dict:
                                    self.class110_dict[(110.2, cur_tag)] = 1
                                else:
                                    self.class110_dict[(110.2, cur_tag)] += 1

                            if (110.1, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.1, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.1, cur_tag)] += 1

                    if re.match('^[0-9\-,.:]*[][0-9]+[0-9\-,.:]*$', cur_word) or (cur_word in ["II", "III", "IV"] or
                                                                                  re.match(
                                                                                      '^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$',
                                                                                      cur_word)):  # CD tag
                        if (110.3, cur_tag) not in self.class110_dict:
                            self.class110_dict[(110.3, cur_tag)] = 1
                        else:
                            self.class110_dict[(110.3, cur_tag)] += 1

                    elif ((re.search('[\-]', cur_word) and
                           (re.search('ing$', cur_word.split('-')[-1]) or
                            re.search('ed$', cur_word.split('-')[-1]) or
                            re.search('ic$', cur_word.split('-')[-1]) or
                            re.search('age$', cur_word.split('-')[-1]) or
                            re.search('like$', cur_word.split('-')[-1]) or
                            re.search('ive$', cur_word.split('-')[-1]) or
                            re.search('ven$', cur_word.split('-')[-1]) or
                            re.search('^pre', cur_word.split('-')[0]) or
                            re.search('^anti', cur_word.split('-')[0]) or
                            re.search('er$', cur_word.split('-')[0]))) or
                          re.search('kDa$', cur_word)) or cur_word in ["CR", "CS"]:  # JJ tag
                        if (110.4, cur_tag) not in self.class110_dict:
                            self.class110_dict[(110.4, cur_tag)] = 1
                        else:
                            self.class110_dict[(110.4, cur_tag)] += 1

                    else:
                        if re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', cur_word) or cur_word == "GCS":  # NNS tag
                            print("-----110.5---" + str(cur_word) + " , " + str(cur_tag))
                            if (110.5, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.5, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.5, cur_tag)] += 1

                        if (re.match('^[A-Z\-0-9.,]+$', cur_word) and cur_word not in ["I", "A", ",", ".", ":"]) or \
                                re.search('[a-z\-][A-Z]', cur_word) or \
                                re.match('^[A-Za-z][\-][a-z]+$', cur_word) or \
                                re.match('^[a-z\-]+[0-9]+$', cur_word) or \
                                re.search('coid$', cur_word) or \
                                re.search('ness$', cur_word):  # NN tag
                            if (110.6, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.6, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.6, cur_tag)] += 1

                        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', cur_word):  # might be JJ tag
                            if (110.7, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.7, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.7, cur_tag)] += 1

                        if re.match('^[A-Z]?[a-z]+[\-][0-9\-.,]+$', cur_word):  # might be NN tag
                            if (110.8, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.8, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.8, cur_tag)] += 1

                        if re.search('\.$', cur_word) and cur_word != ".":  # might be FW tag, but not only
                            if (110.9, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.9, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.9, cur_tag)] += 1

                        if cur_word in ['Treponema', 'cerevisiae', 'pallidum', 'Borrelia', 'burgdorferi',
                                        'vitro', 'vivo', 'i.e.', 'e.g.']:  # FW tag
                            if (110.92, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.92, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.92, cur_tag)] += 1
                        if cur_word in ['in', 'In'] and word_idx != len(splited_words) - 1:  # FW tag
                            next_word = splited_words[word_idx + 1].split('_')[0]
                            if next_word in ['vitro', 'vivo']:
                                if (110.92, cur_tag) not in self.class110_dict:
                                    self.class110_dict[(110.92, cur_tag)] = 1
                                else:
                                    self.class110_dict[(110.92, cur_tag)] += 1

                        if cur_word in ['i', 'ii', 'iii', 'iv']:  # LS tag
                            if (110.93, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.93, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.93, cur_tag)] += 1


class Feature2Id:
    """
    define unique index for each feature which pass its class threshold.
    """

    def __init__(self, feature_statistics: ClassStatistics):
        self.feature_statistics = feature_statistics

        self.all_feature_index_dict = OrderedDict()
        self.n_total_features = 0  # Total number of features accumulated

        self.class100_feature_index_dict = OrderedDict()  # Init all features index dictionaries
        self.n_class100 = 0  # Number of Word\Tag pairs features - this is the index for class 1

        self.class101_feature_index_dict = OrderedDict()
        self.n_class101 = 0

        self.class102_feature_index_dict = OrderedDict()
        self.n_class102 = 0

        self.class103_feature_index_dict = OrderedDict()
        self.n_class103 = 0

        self.class104_feature_index_dict = OrderedDict()
        self.n_class104 = 0

        self.class105_feature_index_dict = OrderedDict()
        self.n_class105 = 0

        self.class106_feature_index_dict = OrderedDict()
        self.n_class106 = 0

        self.class107_feature_index_dict = OrderedDict()
        self.n_class107 = 0

        self.class108_feature_index_dict = OrderedDict()
        self.n_class108 = 0

        self.class109_feature_index_dict = OrderedDict()
        self.n_class109 = 0

        self.class110_feature_index_dict = OrderedDict()
        self.n_class110 = 0

        # TODO delete if no thresholds
        self.suffix_count_dict = OrderedDict()
        self.prefix_count_dict = OrderedDict()

    def set_index_class100(self, threshold=0):
        """
            Extract out of text all word/tag pairs
            :param threshold: feature count threshold - empirical count must be equal or greater than threshold
        """
        for key, value in self.feature_statistics.class100_dict.items():
            if value >= threshold:
                self.class100_feature_index_dict[key] = self.n_class100 + self.n_total_features
                self.n_class100 += 1
        self.n_total_features += self.n_class100

    def set_index_class101(self, threshold=0):
        for key, value in self.feature_statistics.class101_dict.items():
            if value >= threshold:
                self.class101_feature_index_dict[key] = self.n_class101 + self.n_total_features
                self.n_class101 += 1
        self.n_total_features += self.n_class101

    def set_index_class102(self, threshold=0):
        # set thresholds for class, f102 we choose the threshold to be the mean in every length category
        thresholds = dict()  # length of prefix : threshold

        for length in [1, 2, 3, 4, 5, 6, 7]:
            keys = [key for key in self.feature_statistics.class102_dict if len(key[1]) == length]
            values = [self.feature_statistics.class102_dict[key] for key in keys]
            thresholds[length] = np.mean(values)

        for key, value in self.feature_statistics.class102_dict.items():

            if value >= thresholds[len(key[1])]:
                self.class102_feature_index_dict[key] = self.n_class102 + self.n_total_features
                self.n_class102 += 1
        self.n_total_features += self.n_class102

    def set_index_class103(self, threshold=0):
        keys = [key for key in self.feature_statistics.class103_dict]
        values = [self.feature_statistics.class103_dict[key] for key in keys]
        threshold = np.mean(values)

        for key, value in self.feature_statistics.class103_dict.items():
            if value >= threshold:
                self.class103_feature_index_dict[key] = self.n_class103 + self.n_total_features
                self.n_class103 += 1
        self.n_total_features += self.n_class103

    def set_index_class104(self, threshold=0):
        for key, value in self.feature_statistics.class104_dict.items():
            if value >= threshold:
                self.class104_feature_index_dict[key] = self.n_class104 + self.n_total_features
                self.n_class104 += 1
        self.n_total_features += self.n_class104

    def set_index_class105(self, threshold=0):
        for key, value in self.feature_statistics.class105_dict.items():
            if value >= threshold:
                self.class105_feature_index_dict[key] = self.n_class105 + self.n_total_features
                self.n_class105 += 1
        self.n_total_features += self.n_class105

    def set_index_class106(self, threshold=0):
        keys = [key for key in self.feature_statistics.class106_dict]
        values = [self.feature_statistics.class106_dict[key] for key in keys]
        threshold = np.mean(values)

        for key, value in self.feature_statistics.class106_dict.items():
            if value >= threshold:
                self.class106_feature_index_dict[key] = self.n_class106 + self.n_total_features
                self.n_class106 += 1
        self.n_total_features += self.n_class106

    def set_index_class107(self, threshold=0):
        keys = [key for key in self.feature_statistics.class107_dict]
        values = [self.feature_statistics.class107_dict[key] for key in keys]
        threshold = np.mean(values)

        for key, value in self.feature_statistics.class107_dict.items():
            if value >= threshold:
                self.class107_feature_index_dict[key] = self.n_class107 + self.n_total_features
                self.n_class107 += 1
        self.n_total_features += self.n_class107

    def set_index_class108(self, threshold=0):
        for key, value in self.feature_statistics.class108_dict.items():
            if value >= threshold:
                self.class108_feature_index_dict[key] = self.n_class108 + self.n_total_features
                self.n_class108 += 1
        self.n_total_features += self.n_class108

    def set_index_class109(self, threshold=0):
        for key, value in self.feature_statistics.class109_dict.items():
            if value >= threshold:
                self.class109_feature_index_dict[key] = self.n_class109 + self.n_total_features
                self.n_class109 += 1
        self.n_total_features += self.n_class109

    def set_index_class110(self, threshold=0):
        for key, value in self.feature_statistics.class110_dict.items():
            if value >= threshold:
                self.class110_feature_index_dict[key] = self.n_class110 + self.n_total_features
                self.n_class110 += 1
        self.n_total_features += self.n_class110

    def build_all_classes_feature_index_dict(self):
        self.all_feature_index_dict.update(self.class100_feature_index_dict)
        self.all_feature_index_dict.update(self.class101_feature_index_dict)
        self.all_feature_index_dict.update(self.class102_feature_index_dict)
        self.all_feature_index_dict.update(self.class103_feature_index_dict)
        self.all_feature_index_dict.update(self.class104_feature_index_dict)
        self.all_feature_index_dict.update(self.class105_feature_index_dict)
        self.all_feature_index_dict.update(self.class106_feature_index_dict)
        self.all_feature_index_dict.update(self.class107_feature_index_dict)
        self.all_feature_index_dict.update(self.class108_feature_index_dict)
        self.all_feature_index_dict.update(self.class109_feature_index_dict)
        self.all_feature_index_dict.update(self.class110_feature_index_dict)

    # TODO DELETE
    def plots_for_threshold(self):

        # f101
        for length in [1, 2, 3, 4]:
            keys = [key for key in self.feature_statistics.class101_dict if len(key[1]) == length]
            values = [self.feature_statistics.class101_dict[key] for key in keys]
            plt.hist(values, bins=500, range=(0, np.mean(values)))
            plt.title(f'f101 - Suffix - length = {length}')
            plt.show()
            plt.boxplot(values)
            plt.title(f'f101 - Suffix - length = {length}')
            plt.show()

        # f102
        for length in [1, 2, 3, 4]:
            keys = [key for key in self.feature_statistics.class102_dict if len(key[1]) == length]
            values = [self.feature_statistics.class102_dict[key] for key in keys]
            plt.hist(values, bins=500, range=(0, np.mean(values)))
            plt.title(f'f102 - Prefix - length = {length}')
            plt.show()
            plt.boxplot(values)
            plt.title(f'f102 - Prefix - length = {length}')
            plt.show()

        # barplot for specific prefix and all its tags distribution
        # for prefix in ["mill"]:
        #     keys = [key for key in self.feature_statistics.class102_dict if key[1] == prefix]
        #     values = [self.feature_statistics.class102_dict[key] for key in keys]
        #     tags = [key[2] for key in keys]
        #     print(values)
        #     print(tags)
        #     x = np.arange(len(tags))
        #
        #     plt.bar(x, values)
        #     plt.title(f'barplot for specific prefix and all its tags: {prefix}')
        #     plt.xticks(x, tags)
        #     plt.show()

    # TODO DELETE
    def prefix_tags_distribution_condition(self, my_key):
        prefix = my_key[1]

        if prefix not in self.prefix_count_dict:
            prefix_count = 0
            for key, value in self.feature_statistics.class102_dict.items():
                if key[1] == prefix:
                    prefix_count += value
            self.prefix_count_dict.update({prefix: prefix_count})

        if self.feature_statistics.class102_dict[my_key] / self.prefix_count_dict[prefix] > 0.97 and \
                self.feature_statistics.class102_dict[my_key] >= 5:
            return True
        else:
            return False

    # TODO DELETE
    def suffix_tags_distribution_condition(self, my_key):
        suffix = my_key[1]

        if suffix not in self.suffix_count_dict:
            suffix_count = 0
            for key, value in self.feature_statistics.class101_dict.items():
                if key[1] == suffix:
                    suffix_count += value
            self.suffix_count_dict.update({suffix: suffix_count})

        if self.feature_statistics.class101_dict[my_key] / self.suffix_count_dict[suffix] > 0.97 and \
                self.feature_statistics.class101_dict[my_key] >= 5:
            # print(suffix, self.feature_statistics.class101_dict[my_key] / self.suffix_count_dict[suffix])
            return True
        else:
            return False


class ConfusionMatrix:
    def __init__(self, conf_mat: dict, m=None, M=None):
        """
        build ConfusionMatrix object
        :param conf_mat: (true_tag, predicted_tag) keys with values number of occurrences
        :param m: minimum value for color map - everything smaller will get the same color
        :param M: maximum value for color map - everything bigger will get the same color
        """
        self.conf_mat = conf_mat
        self.m = m
        self.M = M

    # source: https://stackoverflow.com/questions/38931566/pandas-style-background-gradient-both-rows-and-columns
    def background_gradient(self, s, low=0, high=0):
        """
        create colors for data frame values
        :param s: data frame
        :param low: parameter for normalization
        :param high: parameter for normalization
        :return: color map to apply
        """
        cmap = sns.light_palette("red", as_cmap=True)
        if self.m is None:
            self.m = s.min().min()
        if self.M is None:
            self.M = s.max().max()
        rng = self.M - self.m
        norm = colors.Normalize(self.m - (rng * low),
                                self.M + (rng * high))
        normed = s.apply(norm)

        cm = plt.cm.get_cmap(cmap)
        c = normed.applymap(lambda x: colors.rgb2hex(cm(x)))
        ret = c.applymap(lambda x: 'background-color: %s' % x)
        return ret

    @staticmethod
    def highlight_green(s):
        color = '#baf1a1'
        return 'background-color: %s' % color

    @staticmethod
    def highlight_zero(s):
        color = '#C2C2C2'
        return 'background-color: %s' % color

    def create_conf_matrix_save_to_html_file(self, output_path: str):
        """
        creates confusion matrix, highlight color and save to .html file
                           TRUE TAG FOR TOP 10 CONFUSED TAGS
                           ________________________________
            PREDICTED TAG  |      ||      ||      ||      |
                           ________________________________
                           |      ||      ||      ||      |
                           ________________________________
                        .
                        .
                        .
                           ________________________________
                           |      ||      ||      ||      |
                           ________________________________

        (in Jupyter / Python Notebook it shows in the interface, in python script need to save to .html to show result)
        :param output_path: the path for .html file to save
        :return: None
        """
        # create dict for wrong tagging in the format ->  true_tag : total amount of mistakes
        conf_matrix_dict_wrong_tagging = dict()
        for (true, pred), amount_mistakes in self.conf_mat.items():
            if true != pred:
                if true not in conf_matrix_dict_wrong_tagging:
                    conf_matrix_dict_wrong_tagging[true] = amount_mistakes
                else:
                    conf_matrix_dict_wrong_tagging[true] += amount_mistakes

        # sort by amount of mistakes desc.
        conf_matrix_dict_wrong_tagging = sorted(conf_matrix_dict_wrong_tagging.items(), key=lambda item: item[1],
                                                reverse=True)

        # see what columns we need for our confusion matrix (columns = true label)
        columns_tags = sorted([true for true, _ in conf_matrix_dict_wrong_tagging[:10]])

        # set rows to be the columns + all other tags (rows = predicted label)
        rows_tags = list(columns_tags) + sorted(set([true_tag for (true_tag, pred_tag), _ in self.conf_mat.items() if
                                                     true_tag not in columns_tags]))

        # create empty DataFrame with rows and cols needed
        df = pd.DataFrame(0, index=rows_tags, columns=columns_tags)

        # fill the DataFrame with the values
        for pred_tag in df.index:
            for true_tag in df.columns:
                if (true_tag, pred_tag) in self.conf_mat:
                    df.loc[pred_tag][true_tag] = self.conf_mat[(true_tag, pred_tag)]

                else:
                    df.loc[pred_tag][true_tag] = 0

        # change colors for wrong tagging
        style = df.style.apply(self.background_gradient, axis=None)

        # change color for 0 values
        for pred_tag in df.index:
            for true_tag in df.columns:
                if df.loc[pred_tag][true_tag] == 0:
                    style = style.applymap(self.highlight_zero, subset=pd.IndexSlice[pred_tag, true_tag])

        # change color for correct tagging
        for tag in columns_tags:
            style = style.applymap(self.highlight_green, subset=pd.IndexSlice[tag, tag])

        # write the pandas styler object to HTML -> can view the confusion matrix from browser
        with open(output_path, "w") as html:
            html.write('<font size="10" face="Courier New" >' + style.render() + '</font>')


# Auxiliary functions for class f108
def re_match_words(regular_exp: str, lst):
    if not [w for w in lst if w != '']:
        return False
    for word in lst:
        if word != '' and not re.match(regular_exp, word):
            return False
    return True


def re_match_letters_numbers(regular_exps: list, lst):
    for i, word in enumerate(lst):
        if i % 2 == 0:
            if word != '' and not re.match(regular_exps[i % 2], word):
                return False
        else:
            if word != '' and not re_match_words(regular_exps[i % 2], re.split('[,]|[:]|[\\\\]|[/]|[%]', word)):
                return False
    return True


def re_match_numbers_letters(regular_exps: list, lst):
    for i, word in enumerate(lst):
        if i % 2 == 0:
            if word != '' and not re_match_words(regular_exps[i % 2], re.split('[,]|[:]|[\\\\]|[/]|[%]', word)):
                return False
        else:
            if word != '' and not re.match(regular_exps[i % 2], word):
                return False
    return True


"""START OF SECTION FOR OBJECTIVE AND GRADIENT CALC"""


def my_sum(obj):
    if type(obj) == np.float64:
        return obj
    if type(obj) == np.ndarray:
        if not obj.size > 0:
            return 0
    return sum(obj)


# global variables for objective function optimization
first_iteration = 1
active_indices_sentences_i = 0
flattened_f_xi_yi = 0
gradient_left_sigma = 0
all_active_indices = 0
softmax_indices_for_gradient = 0


def function_l_and_gradient_l_special(v: np.array, *args):
    file_path = args[0]
    lam = args[1]
    feature_statistics = args[2]
    features_indices = args[3]

    # first part - saving all the needed 'f' values in the first iteration only (one time per train file)
    global active_indices_sentences_i
    global flattened_f_xi_yi
    global gradient_left_sigma
    global all_active_indices
    global first_iteration
    global softmax_indices_for_gradient
    sentences = sum(1 for _ in open(file_path))

    if first_iteration:
        start = time.time()
        first_iteration = 0

        # global variables defined in global scope
        active_indices_sentences_i = [list() for _ in range(sentences)]  # true tag that really seen in tag
        flattened_f_xi_yi = []  # all true tags flattened
        gradient_left_sigma = np.zeros(len(v))
        all_active_indices = [list() for _ in range(sentences)]  # every tag for every word in every sentence
        # all_active_indices = [[list for every sentence]] -> :
        # -> list of a sentence: [dict for i=0 {}, dict for i=1,....] (i = 0,...,num of words - 1)
        # -> dict looks like {key = 'tag', value: [active indices]}

        s = 0  # running count for lines processed
        with open(file_path) as f:
            for line in f:
                words, tags = split_sentence_to_words_and_tags(line)
                for i in range(len(words)):
                    active = f_xi_yi(features_indices, words, tags, i)
                    active_indices_sentences_i[s].append(active)
                    gradient_left_sigma[active] += 1
                    all_active_indices[s].append(dict())
                    t_i = tags[i]
                    for y in feature_statistics.Y:
                        tags[i] = y
                        all_active_indices[s][i][y] = f_xi_yi(features_indices, words, tags, i)
                    tags[i] = t_i
                s += 1

        # used to calculate easily the left sigma in objective function
        flattened_f_xi_yi = [value for list1 in active_indices_sentences_i for list2 in list1 for value in list2]

        # used to calculate the right sigma of gradient
        softmax_indices_for_gradient = [list() for _ in range(len(v))]
        cnt_words = 0
        for s in range(sentences):
            for i in range(len(all_active_indices[s])):
                for y in range(len(feature_statistics.Y)):
                    for index in all_active_indices[s][i][feature_statistics.Y[y]]:
                        softmax_indices_for_gradient[index].append((cnt_words, y))
                cnt_words += 1

        stop = time.time()
        print(f"the data prep took: {stop - start} secs")
        # finished collecting all data

    start = time.time()
    # updates we need to make according to current 'v'
    # ------ OBJECTIVE ------
    objective_value = sum(v[flattened_f_xi_yi])  # left sigma objective (all sentences)
    exponents = [
        [my_sum(v[dict_indices]) for dict_indices in all_active_indices[s][i].values()] for s in range(sentences) for i
        in range(len(all_active_indices[s]))]
    logsumexp = special.logsumexp(exponents, axis=1, keepdims=True)
    objective_value -= np.float64(sum(logsumexp))  # we subtract the right sigma (all sentences)

    # ------ GRADIENT ------
    gradient_value = np.array(gradient_left_sigma)  # left sigma gradient (all sentences)
    gradient_right_sigma = np.zeros(len(v))  # (all sentences)
    softmax = np.exp(exponents - logsumexp)  # line from special.softmax documentation
    # calculate right gradient sigma
    for k in range(len(v)):
        for indices in softmax_indices_for_gradient[k]:
            gradient_right_sigma[k] += softmax[indices[0]][indices[1]]
    gradient_value -= gradient_right_sigma

    stop = time.time()
    print(f"the iteration took: {stop - start} secs")
    print(f'objective value = {(-1 * (objective_value - (lam / 2) * (np.linalg.norm(v)) ** 2))}')
    print(f'gradient norm {np.linalg.norm(gradient_value - lam * v)}')

    return -1 * (objective_value - (lam / 2) * (np.linalg.norm(v) ** 2)), -1 * (gradient_value - lam * v)


"""END OF SECTION OBJECTIVE AND GRADIENT CALC"""


def f_xi_yi(features_indices: Feature2Id, words, tags, i):
    active_features_indices = list()
    cur_word = words[i]
    prev_word = words[i - 1] if i > 0 else '*'
    next_word = words[i + 1] if i < len(words) - 1 else 'STOP'
    cur_tag = tags[i]
    prev_tag = tags[i - 1] if i > 0 else ''
    prev_prev_tag = tags[i - 2] if i - 1 > 0 else ''

    # features fired in class 100
    if (100, cur_word, cur_tag) in features_indices.class100_feature_index_dict:
        active_features_indices.append(features_indices.all_feature_index_dict[(100, cur_word, cur_tag)])
    # elif (100, cur_word.lower(), cur_tag) in features_indices.class100_feature_index_dict:
    #     active_features_indices.append(features_indices.all_feature_index_dict[(100, cur_word.lower(), cur_tag)])
    # elif (100, cur_word.upper(), cur_tag) in features_indices.class100_feature_index_dict:
    #     active_features_indices.append(features_indices.all_feature_index_dict[(100, cur_word.upper(), cur_tag)])
    elif (100, cur_word[0].upper() + cur_word[1:].lower(), cur_tag) in features_indices.class100_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(100, cur_word[0].upper() + cur_word[1:].lower(), cur_tag)])

    # features fired in class 101
    n = min(len(cur_word) - 1, 7)
    for suffix_length in range(1, n + 1):
        if (101, cur_word[-suffix_length:], cur_tag) in features_indices.class101_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(101, cur_word[-suffix_length:], cur_tag)])

    # features fired in class 102
    for prefix_length in range(1, n + 1):
        if (102, cur_word[:prefix_length], cur_tag) in features_indices.class102_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(102, cur_word[:prefix_length], cur_tag)])

    # features fired in class 103

    if (103, prev_prev_tag, prev_tag, cur_tag) in features_indices.class103_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(103, prev_prev_tag, prev_tag, cur_tag)])

    # features fired in class 104

    if (104, prev_tag, cur_tag) in features_indices.class104_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(104, prev_tag, cur_tag)])

    # features fired in class 105
    if (105, cur_tag) in features_indices.class105_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(105, cur_tag)])

    # features fired in class 106
    if (106, prev_word, cur_tag) in features_indices.class106_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(106, prev_word, cur_tag)])

    # features fired in class 107

    if (107, next_word, cur_tag) in features_indices.class107_feature_index_dict:
        active_features_indices.append(
            features_indices.all_feature_index_dict[(107, next_word, cur_tag)])

    # features fired in class 108
    hyphen_count = cur_word.count('-')
    if re.search(r'\d', cur_word):
        splited_cur_word = re.split('-', cur_word)
        no_category = True
        if re_match_words('^[0-9]+$', re.split('[-]|[,]|[.]|[:]|[\\\\]|[/]|[%]', cur_word)):
            if (108.1, cur_tag) in features_indices.class108_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(108.1, cur_tag)])
            no_category = False

        elif hyphen_count > 1 and re_match_letters_numbers(['^[A-Za-z]+$', '^[0-9]+$'],
                                                           splited_cur_word) \
                and not cur_word.startswith('mid'):
            if re.match('^[A-Za-z]+$', splited_cur_word[-1]):
                if (108.21, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(108.21, cur_tag)])
            else:
                if (108.22, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(108.22, cur_tag)])
            no_category = False

        elif hyphen_count == 1 and re_match_letters_numbers(['^[A-Za-z]+$', '^[0-9]+$'],
                                                            splited_cur_word):
            if re.match('^[a-z]$', cur_word[0]):
                if (108.31, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(108.31, cur_tag)])
            else:
                if (108.32, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(108.32, cur_tag)])

            no_category = False

        elif hyphen_count > 1 and re_match_numbers_letters(['^[0-9]+$', '^[A-Za-z]+$'],
                                                           splited_cur_word):
            if (108.4, cur_tag) in features_indices.class108_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(108.4, cur_tag)])
            no_category = False

        elif hyphen_count == 1 and re_match_numbers_letters(['^[0-9]+$', '^[A-Za-z]+$'],
                                                            splited_cur_word):
            if (108.5, cur_tag) in features_indices.class108_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(108.5, cur_tag)])
            no_category = False

        if re.match('^([a-zA-Z]*(-)?[0-9][0-9][0-9][0-9](s?))$|^(\'[0-9][0-9]s)$', cur_word):
            if (108.6, cur_tag) in features_indices.class108_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(108.6, cur_tag)])
            no_category = False

        if no_category:
            if re.match('^[A-Za-z]+$', splited_cur_word[-1]):
                if (108.7, hyphen_count, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(
                        features_indices.all_feature_index_dict[(108.7, hyphen_count, cur_tag)])
            else:
                if (108.8, hyphen_count, cur_tag) in features_indices.class108_feature_index_dict:
                    active_features_indices.append(
                        features_indices.all_feature_index_dict[(108.8, hyphen_count, cur_tag)])

    # features fired in class 109
    if i > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word) \
            and re.match('^[A-Z]', prev_word):
        if (109.1, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.1, prev_tag, cur_tag)])

    elif i > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', prev_word):
        if (109.2, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.2, prev_tag, cur_tag)])

    elif i > 0 and re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word):

        if (109.3, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.3, prev_tag, cur_tag)])

    elif i > 0 and re.match('^[A-Z]', cur_word) and re.match('^[A-Z]', next_word) \
            and re.match('^[A-Z]', prev_word):
        if (109.4, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.4, prev_tag, cur_tag)])

    elif i == 0 or (i > 0 and prev_word in ['``', '.']):
        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word) \
                and re.match('^[A-Z]', prev_word):
            if (109.5, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
                active_features_indices.append(
                    features_indices.all_feature_index_dict[(109.5, prev_tag, cur_tag)])

    elif i == 0 or (i > 0 and prev_word in ['``', '.']):
        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word) and re.match('^[A-Z]', next_word):
            if (109.6, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
                active_features_indices.append(
                    features_indices.all_feature_index_dict[(109.6, prev_tag, cur_tag)])

    elif i == 0 or (i > 0 and prev_word in ['``', '.']):
        if re.match('^[A-Z](.*?)[-.]+(.*?)', cur_word):
            if (109.7, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
                active_features_indices.append(
                    features_indices.all_feature_index_dict[(109.7, prev_tag, cur_tag)])

    elif i == 0 or (i > 0 and prev_word in ['``', '.']):
        if re.match('^[A-Z]', cur_word) and re.match('^[A-Z]', next_word) \
                and re.match('^[A-Z]', prev_word):
            if (109.8, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
                active_features_indices.append(
                    features_indices.all_feature_index_dict[(109.8, prev_tag, cur_tag)])

    elif re.match('^[A-Z][A-Z]+$', cur_word) and re.match('^[A-Z][A-Z]+$', next_word) and re.match(
            '^[A-Z][A-Z]+$', prev_word):
        if (109.9, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.9, prev_tag, cur_tag)])

    elif re.match('^[A-Z][A-Z]+$', cur_word) and re.match('^[A-Z][A-Z]+$', next_word):
        if (109.11, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.11, prev_tag, cur_tag)])

    elif re.match('^[A-Z][A-Z]+$', cur_word):
        if (109.12, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.12, prev_tag, cur_tag)])

    if re.match('[A-Z]', cur_word):
        if (109.13, hyphen_count, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.13, hyphen_count, prev_tag, cur_tag)])

    if re.match('(.*?)[a-z](.*?)[A-Z]', cur_word):
        if (109.14, prev_tag, cur_tag) in features_indices.class109_feature_index_dict:
            active_features_indices.append(
                features_indices.all_feature_index_dict[(109.14, prev_tag, cur_tag)])

    # features fired in class 110
    if len(words[i]) >= 13:
        if re.search('ing$', words[i]):
            if (110.11, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.11, tags[i])])
        elif re.search('ed$', words[i]):
            if (110.14, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.14, tags[i])])

        elif re.search('ally$', words[i]) or re.search('ely$', words[i]) or re.search('tly$', words[i]):  # RB tag
            if (110.12, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.12, tags[i])])

        elif re.search('tant$', words[i]) or \
                re.search('cal$', words[i]) or \
                re.search('ic$', words[i]) or \
                re.search('ive$', words[i]) or \
                re.search('nal$', words[i]) or \
                re.search('-dependent$', words[i]) or \
                re.search('-sensitive$', words[i]) or \
                re.search('-specific$', words[i]) or \
                re.search('tly$', words[i]):  # JJ tag
            if (110.13, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.13, tags[i])])
        else:  # NN + NNS tags
            if words[i][-1] == 's':
                if (110.2, tags[i]) in features_indices.class110_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(110.2, tags[i])])

            if (110.1, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.1, tags[i])])

    if re.match('^[0-9\-,.:]*[0-9]+[0-9\-,.:]*$', words[i]) or words[i] in ["II", "III", "IV"] or \
            re.match('^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$', words[i]):  # CD tag
        if (110.3, tags[i]) in features_indices.class110_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(110.3, tags[i])])

    elif ((re.search('[\-]', words[i]) and
           (re.search('ing$', words[i].split('-')[-1]) or
            re.search('ed$', words[i].split('-')[-1]) or
            re.search('ic$', words[i].split('-')[-1]) or
            re.search('age$', words[i].split('-')[-1]) or
            re.search('like$', words[i].split('-')[-1]) or
            re.search('ive$', words[i].split('-')[-1]) or
            re.search('ven$', words[i].split('-')[-1]) or
            re.search('^pre', words[i].split('-')[0]) or
            re.search('^anti', words[i].split('-')[0]) or
            re.search('er$', words[i].split('-')[0]))) or
          re.search('kDa$', words[i])) or words[i] in ["CR", "CS"]:  # JJ tag
        if (110.4, tags[i]) in features_indices.class110_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(110.4, tags[i])])
    else:
        if re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', words[i]) or words[i] == "GCS":  # NNS tag
            if (110.5, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.5, tags[i])])

        if (re.match('^[A-Z\-0-9.,]+$', words[i]) and words[i] not in ["I", "A", ",", ".", ":"]) or \
                re.search('[a-z][A-Z]', words[i]) or \
                re.match('[A-Za-z][\-][A-Za-z0-9.,\-]+$', words[i]) or \
                re.search('coid$', words[i]) or re.search('ness$', words[i]):  # NN tag
            if (110.6, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.6, tags[i])])

        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', words[i]):  # might be JJ tag
            if (110.7, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.7, tags[i])])

        if re.match('^[A-Z]?[a-z]+[\-][0-9\-.,]+$', words[i]):  # might be NN tag.
            if (110.8, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.8, tags[i])])

        if re.search('\.$', words[i]) and words[i] != ".":  # might be FW tag, but maybe more
            if (110.9, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.9, tags[i])])

        if words[i] in ['Treponema', 'pallidum', 'Borrelia', 'burgdorferi', 'vitro', 'vivo']:  # FW tag
            if (110.92, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.92, tags[i])])

        if words[i] in ['in', 'In'] and i != len(words) - 1:  # FW tag
            next_word = words[i + 1]
            if next_word in ['vitro', 'vivo']:
                if (110.92, tags[i]) in features_indices.class110_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(110.92, tags[i])])

        if words[i] in ['i', 'ii', 'iii', 'iv']:  # LS tag
            if (110.93, tags[i]) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.93, tags[i])])

    return active_features_indices


def split_sentence_to_words_and_tags(line: str):
    """
    Split the sentence with words and tags into 2 lists
    :param line: input line from text file. e.g. "'The_DT Treasury_NNP is_VBZ still_RB ._.\n"
    :return: [list of words in order], [list of tags in order]
    """
    splited_words = re.split(' |[\n]', line)
    if splited_words[-1] == "":
        del splited_words[-1]  # remove \n

    words, tags = list(), list()
    for word_tag_str in splited_words:
        word_tag_list = word_tag_str.split('_')
        words.append(word_tag_list[0])
        tags.append(word_tag_list[1])

    return words, tags


def q_params_aux(words, t, u, v, weights, features_indices):
    # Notice that len(words) != len(tags) , but its ok
    return sum(weights[f_xi_yi(features_indices, words=words, tags=[t, u, v], i=2)])


def q_params_calc(k, t, u, weights, features_indices: Feature2Id, words: list, class_statistics: ClassStatistics):
    Y = list(class_statistics.Y)

    # set words list according to current k
    if k == 0:
        try:
            tmp_words = ['*', '*', words[k], words[k + 1]]
        except IndexError:
            tmp_words = ['*', '*', words[k], 'STOP']
    elif k == 1:
        try:
            tmp_words = ['*', words[k - 1], words[k], words[k + 1]]
        except IndexError:
            tmp_words = ['*', words[k - 1], words[k], 'STOP']
    elif k == len(words) - 1:
        tmp_words = [words[k - 2], words[k - 1], words[k], 'STOP']
    else:
        tmp_words = [words[k - 2], words[k - 1], words[k], words[k + 1]]

    softmax = special.softmax([q_params_aux(tmp_words, t, u, v, weights, features_indices) for v in Y])

    q_params = dict()
    for i in range(len(softmax)):
        q_params[Y[i]] = softmax[i]

    return q_params


def beam_search(B, candidates_dict):
    """
    :param B: the beam search parameter >= 1 . this is the number of elements to return.
    :param candidates_dict: keys are (k, u, v) and values are the probabilities. we are search for the B keys of the
    highest probabilities.
    :return: list of B tuples [(k, u1, v1), (k, u2, v2)...]
    """
    chosen_B_keys = set()  # if change to list, doesn't suppose to have problem. we expect unique values only
    for key, value in {k: v for k, v in
                       sorted(candidates_dict.items(), key=lambda item: item[1], reverse=True)}.items():
        chosen_B_keys.add(key)
        if len(chosen_B_keys) == B:
            return list(chosen_B_keys)

    return list(chosen_B_keys)


def memm_viterbi(weights, features_indices: Feature2Id, words: list, class_statistics: ClassStatistics, beam):
    """
    :param class_statistics:
    :param weights: chosen weights vector. will not be changed while the viterbi algorithm
    :param features_indices:
    :param words: a sentence to tag
    :param beam: number for beam search e.g: np.inf , 5, 10
    :return: inference of tags sequence
    """
    # TODO should determine last tag to be '.' and '"' only? there are samples which other tag?
    # TODO should decide deterministic for special values, e.g. ('-RRB-', '``', '$', '#', ':', ',', '.', "''",'-LRB-')
    Y = class_statistics.Y

    tags_infer = [None] * len(words)

    # pi = {key: (k, u, v) : value = max prob a tag sequence ending in tags u, v at positions k-1, k}
    pi = dict()
    pi[(-1, "*", "*")] = 1

    # bp = {key: (k, u, v) : value: the t argmax of the corresponding key in pi dict. t is in position k-2}
    bp = dict()

    before_u = dict({"*": ["*"]})
    q_params = dict()  # {(k, t, u): {v: softmax result} }

    for k in range(0, len(words)):
        pi_temporary, bp_temporary = dict(), dict()

        for u, list_of_ts in before_u.items():
            for v in Y:
                max_t = -np.inf
                argmax_t = None
                for t in list_of_ts:
                    if (k, t, u) not in q_params.keys():
                        q_params[(k, t, u)] = q_params_calc(k, t, u, weights, features_indices, words,
                                                            class_statistics)
                    current = pi[(k - 1, t, u)] * q_params[(k, t, u)][v]
                    if current > max_t:
                        max_t = current
                        argmax_t = t
                if max_t != -np.inf:
                    pi_temporary[(k, u, v)] = max_t
                    bp_temporary[(k, u, v)] = argmax_t
            chosen_B_keys = beam_search(beam, candidates_dict=pi_temporary)  # contains <=B elements like (k, u, v)
            before_u = dict()
            for key in chosen_B_keys:
                pi[key] = pi_temporary[key]
                bp[key] = bp_temporary[key]
                if key[2] in before_u:
                    before_u[key[2]].append(key[1])
                else:
                    before_u[key[2]] = list()
                    before_u[key[2]].append(key[1])

    # set (t_n-1, t_n)
    argmax_u_v = [None, None]
    max_u_v = -np.inf
    for v, list_of_us in before_u.items():  # suppose to have only one v ('.' or '"')
        for u in list_of_us:
            current = pi[(len(words) - 1, u, v)]
            if current > max_u_v:
                max_u_v = current
                argmax_u_v = [u, v]  # tags in two last places

    tags_infer[len(words) - 1] = argmax_u_v[1]
    tags_infer[len(words) - 2] = argmax_u_v[0]

    for k in range(len(words) - 3, -1, -1):
        tags_infer[k] = bp[(k + 2, tags_infer[k + 1], tags_infer[k + 2])]

    # Deterministic tagging  # TODO Determinitic tagging. maybe can do it smarter and change the Y that v is running on
    for i in range(len(words)):
        if words[i] in [";", "--"]:
            tags_infer[i] = ":"

    return tags_infer


def generate_inference_file(path_file_to_tag: str, path_result: str, weights, features_indices: Feature2Id,
                            class_statistics: ClassStatistics, beam):
    """
    create a tagged POS file for the given file to tag
    :param path_file_to_tag: file to tag with sentences lines (can be tagged or not tagged)
    :param path_result: path to write the result file to
    :param weights: model weights
    :param features_indices: relevant Feature2Id object
    :param class_statistics: relevant ClassStatistics object
    :param beam: number for beam search e.g: np.inf , 5, 10
    :return: None
    """
    with open(path_result, 'w') as write_file:
        with open(path_file_to_tag) as read_file:
            for line in read_file:
                start = time.time()
                # prepare words list of 'line'
                splited_words = re.split(' |[\n]', line)
                last_char = ""  # for the last line in the file
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                    last_char = '\n'
                words = [w.split('_')[0] for w in splited_words]
                # get the predicted tags for the sentence
                tags = memm_viterbi(weights=weights, features_indices=features_indices, words=words,
                                    class_statistics=class_statistics, beam=beam)
                # build the line to write and write it to result path
                line_to_write = ' '.join(['_'.join(word_tag) for word_tag in list(zip(words, tags))]) + last_char
                write_file.write(line_to_write)
                stop = time.time()
                # print(f'tagging of sentence: {line}\n       took {stop - start} secs') # TODO Delete


def compare_tagged_files(path_true: str, path_predicted: str, class_statistics: ClassStatistics) -> (dict, float):
    """
    prepare a dictionary for (true_tag, predicted_tag) keys with values number of occurrences,
    and calculate tagging accuracy
    :param path_true: path for tagged file with true labels
    :param path_predicted: path for tagged file with predicted labels
    :param class_statistics: relevant ClassStatistics object
    :return: dict, float as described
    """
    correct, wrong = 0, 0
    # create dict for the confusion matrix -> {(true_tag, pred_tag) : number of occurrences}
    dict_for_confusion_matrix = {(t1, t2): 0 for t1 in class_statistics.Y for t2 in class_statistics.Y}

    true_sentences = [s for s in open(path_true)]
    predicted_sentences = [s for s in open(path_predicted)]

    for s in range(len(true_sentences)):
        splited_true = re.split(' |[\n]', true_sentences[s])
        splited_predicted = re.split(' |[\n]', predicted_sentences[s])

        if splited_true[-1] == "":
            del splited_true[-1]  # remove \n

        if splited_predicted[-1] == "":
            del splited_predicted[-1]  # remove \n

        # TODO DELETE
        # tags_true = [w.split('_')[1] for w in splited_true]
        # tags_predicted = [w.split('_')[1] for w in splited_predicted]
        for true, predicted in zip(splited_true, splited_predicted):
            if true.split('_')[1] != predicted.split('_')[1]:
                wrong += 1
                # print(true.split('_')[0], true.split('_')[1], predicted.split('_')[1])
            else:
                correct += 1

            if (true.split('_')[1], predicted.split('_')[1]) in dict_for_confusion_matrix:
                dict_for_confusion_matrix[(true.split('_')[1], predicted.split('_')[1])] += 1
            else:  # for the case where we have some tag in the test that we didn't see during train
                dict_for_confusion_matrix[(true.split('_')[1], predicted.split('_')[1])] = 1

        #  TODO Replace back to original code:
        # tags_true = [w.split('_')[1] for w in splited_true]
        # tags_predicted = [w.split('_')[1] for w in splited_predicted]
        # for true, predicted in zip(tags_true, tags_predicted):
        #     if true != predicted:
        #         wrong += 1
        #     else:
        #         correct += 1
        #
        #     if (true, predicted) in dict_for_confusion_matrix:
        #         dict_for_confusion_matrix[(true, predicted)] += 1
        #     else:  # for the case where we have some tag in the test that we didn't see during train
        #         dict_for_confusion_matrix[(true, predicted)] = 1

    return dict_for_confusion_matrix, float(correct / (correct + wrong))


def split_train_test(tagged_file_path: str, train_path: str, test_path: str, test_size=25):
    """
    split a train file to: train set, validation set (randomlly)
    :param tagged_file_path: a path for a tagged train file
    :param train_path: path (e.g. train.txt) to save the train file
    :param test_path: path (e.g. test.txt) to save the test file
    :param test_size: the amount of sentences to put aside for validation, all the rest goes to train
    :return: the path for the 2 files saved
    """
    # prepare sentences list from tagged file
    with open(tagged_file_path) as file:
        sentences = [line for line in file]
    np.random.shuffle(sentences)

    cur_time = time.time()

    # write test lines to file
    with open(f'{cur_time}_{test_path}', 'w') as file:
        for line in sentences[:test_size]:
            file.write(line)

    # write train lines to file
    with open(f'{cur_time}_{train_path}', 'w') as file:
        for line in sentences[test_size:]:
            file.write(line)

    return f'{cur_time}_{train_path}', f'{cur_time}_{test_path}'


def run_model_1():
    start = time.time()
    global first_iteration
    global active_indices_sentences_i
    global flattened_f_xi_yi
    global gradient_left_sigma
    global all_active_indices
    global softmax_indices_for_gradient
    first_iteration = 1
    active_indices_sentences_i = 0
    flattened_f_xi_yi = 0
    gradient_left_sigma = 0
    all_active_indices = 0
    softmax_indices_for_gradient = 0

    train_file_path = r'train1.wtag'
    test_file_path = r'test1.wtag'

    factr = 1e11
    lambd = 0.2
    beam = 50
    run_optimization = True
    result_file_path = f'{test_file_path}_tagged_{time.time()}'
    info = "THRESHOLD: 102, 103, 106, 107 np.mean()\n" \
           "length prefix and suffix = 7.\n" \
           "f_xi_yi without 2 middle conditions\n"

    c = ClassStatistics(train_file_path)
    c.set_class100_dict()
    c.set_class101_dict()
    c.set_class102_dict()
    c.set_class103_dict()
    c.set_class104_dict()
    c.set_class105_dict()
    c.set_class106_dict()
    c.set_class107_dict()
    c.set_class108_dict()
    c.set_class109_dict()
    c.set_class110_dict()

    f = Feature2Id(c)
    f.set_index_class100()
    f.set_index_class101()
    f.set_index_class102()
    f.set_index_class103()
    f.set_index_class104()
    f.set_index_class105()
    f.set_index_class106()
    f.set_index_class107()
    f.set_index_class108()
    f.set_index_class109()
    f.set_index_class110()

    print(f'f100 features amount = {len(f.class100_feature_index_dict)}')
    print(f'f101 features amount = {len(f.class101_feature_index_dict)}')
    print(f'f102 features amount = {len(f.class102_feature_index_dict)}')
    print(f'f103 features amount = {len(f.class103_feature_index_dict)}')
    print(f'f104 features amount = {len(f.class104_feature_index_dict)}')
    print(f'f105 features amount = {len(f.class105_feature_index_dict)}')
    print(f'f106 features amount = {len(f.class106_feature_index_dict)}')
    print(f'f107 features amount = {len(f.class107_feature_index_dict)}')
    print(f'f108 features amount = {len(f.class108_feature_index_dict)}')
    print(f'f109 features amount = {len(f.class109_feature_index_dict)}')
    print(f'f110 features amount = {len(f.class110_feature_index_dict)}')

    # print(f'f108 features  = {c.class108_dict}')
    # print(f'f109 features  = {c.class109_dict}')
    sorted_to_print = {k: v for k, v in sorted(c.class110_dict.items(), key=lambda item: item[1], reverse=True)}
    print(f'f110 features  = {sorted_to_print}')

    f.build_all_classes_feature_index_dict()

    # TODO choose pgtol, factr, lamb
    # args= [file_path, lambda , feature_statistics, features_indices]
    args = (train_file_path, lambd, c, f)
    x0 = np.random.randn(f.n_total_features)

    v_star_file_name = f'v_star_file={train_file_path}_lam={lambd}_factr={factr}_time={time.time()}.pkl'

    print(
        f"lambda= {lambd},\n"
        f"factr= {factr},\n"
        f"train_file= {train_file_path},\n"
        f"test_file= {test_file_path},\n"
        f"x0 length= {len(x0)}\n"
        f"beam= {beam}\n"
        f"info= {info}\n"
        f"v_star_file_name = {v_star_file_name}\n"
    )

    if run_optimization:
        if type(factr) != str:  # factr is a number
            optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l_special, x0=x0, args=args,
                                                    factr=factr, disp=1)
        else:  # factr is a string = 'default'
            optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l_special, x0=x0, args=args,
                                                    disp=1)

        v_star = optimal_params[0]
        with open(v_star_file_name, 'wb') as file:
            pickle.dump(optimal_params[0], file)

    else:
        """LOAD PICKLE"""
        v_star = pickle.load(open(r'', "rb"))

    stop = time.time()
    print(f'MODEL 1 TRAINING TOOK {stop - start} secs')

    generate_inference_file(path_file_to_tag=test_file_path, path_result=result_file_path, weights=v_star,
                            features_indices=f, class_statistics=c, beam=beam)

    conf_mat_dict, accuracy = compare_tagged_files(path_true=test_file_path, path_predicted=result_file_path,
                                                   class_statistics=c)
    print(f'\nconf_mat_dict = {conf_mat_dict}')
    print(f'\naccuracy = {accuracy}')

    # conf_mat = ConfusionMatrix(conf_mat_dict, m=1, M=50)
    # conf_mat.create_conf_matrix_save_to_html_file(f'conf_mat_{time.time()}.html')


def run_model_2():
    accuracies = list()
    number_of_runs = 50
    for _ in range(number_of_runs):
        global first_iteration
        global active_indices_sentences_i
        global flattened_f_xi_yi
        global gradient_left_sigma
        global all_active_indices
        global softmax_indices_for_gradient
        first_iteration = 1
        active_indices_sentences_i = 0
        flattened_f_xi_yi = 0
        gradient_left_sigma = 0
        all_active_indices = 0
        softmax_indices_for_gradient = 0

        train_path, test_path = split_train_test('train2.wtag', 'train2.txt', 'test2.txt', test_size=25)  # 90% / 10%
        train_file_path = train_path
        test_file_path = test_path
        factr = 'default'
        lambd = 0.02
        beam = 50
        run_optimization = True
        result_file_path = f'{test_file_path}_tagged_{time.time()}'
        info = "THRESHOLD: 102, 103, 106, 107 np.mean()\n" \
               "length prefix and suffix = 7.\n" \
                "f_xi_yi without 2 middle conditions\n"

        c = ClassStatistics(train_file_path)
        c.set_class100_dict()
        c.set_class101_dict()
        c.set_class102_dict()
        c.set_class103_dict()
        c.set_class104_dict()
        c.set_class105_dict()
        c.set_class106_dict()
        c.set_class107_dict()
        c.set_class108_dict()
        c.set_class109_dict()
        c.set_class110_dict()

        f = Feature2Id(c)
        f.set_index_class100()
        f.set_index_class101()
        f.set_index_class102()
        f.set_index_class103()
        f.set_index_class104()
        f.set_index_class105()
        f.set_index_class106()
        f.set_index_class107()
        f.set_index_class108()
        f.set_index_class109()
        f.set_index_class110()

        print(f'f100 features amount = {len(f.class100_feature_index_dict)}')
        print(f'f101 features amount = {len(f.class101_feature_index_dict)}')
        print(f'f102 features amount = {len(f.class102_feature_index_dict)}')
        print(f'f103 features amount = {len(f.class103_feature_index_dict)}')
        print(f'f104 features amount = {len(f.class104_feature_index_dict)}')
        print(f'f105 features amount = {len(f.class105_feature_index_dict)}')
        print(f'f106 features amount = {len(f.class106_feature_index_dict)}')
        print(f'f107 features amount = {len(f.class107_feature_index_dict)}')
        print(f'f108 features amount = {len(f.class108_feature_index_dict)}')
        print(f'f109 features amount = {len(f.class109_feature_index_dict)}')
        print(f'f110 features amount = {len(f.class110_feature_index_dict)}')

        # print(f'f108 features  = {c.class108_dict}')
        # print(f'f109 features  = {c.class109_dict}')
        sorted_to_print = {k: v for k, v in sorted(c.class110_dict.items(), key=lambda item: item[1], reverse=True)}
        print(f'f110 features  = {sorted_to_print}')

        f.build_all_classes_feature_index_dict()

        # TODO choose pgtol, factr, lamb
        # args= [file_path, lambda , feature_statistics, features_indices]
        args = (train_file_path, lambd, c, f)
        x0 = np.random.randn(f.n_total_features)

        v_star_file_name = f'v_star_file={train_file_path}_lam={lambd}_factr={factr}_time={time.time()}.pkl'

        print(
            f"lambda= {lambd},\n"
            f"factr= {factr},\n"
            f"train_file= {train_file_path},\n"
            f"test_file= {test_file_path},\n"
            f"x0 length= {len(x0)}\n"
            f"beam= {beam}\n"
            f"info= {info}\n"
            f"v_star_file_name = {v_star_file_name}\n"
        )

        if run_optimization:
            if type(factr) != str:  # factr is a number
                optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l_special, x0=x0, args=args,
                                                        factr=factr, disp=1)
            else:  # factr is a string = 'default'
                optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l_special, x0=x0, args=args,
                                                        disp=1)

            v_star = optimal_params[0]
            with open(v_star_file_name, 'wb') as file:
                pickle.dump(optimal_params[0], file)

        else:
            """LOAD PICKLE"""
            v_star = pickle.load(open(r'', "rb"))

        generate_inference_file(path_file_to_tag=test_file_path, path_result=result_file_path, weights=v_star,
                                features_indices=f, class_statistics=c, beam=beam)

        conf_mat_dict, accuracy = compare_tagged_files(path_true=test_file_path, path_predicted=result_file_path,
                                                       class_statistics=c)
        accuracies.append(accuracy)
        print(f'\nconf_mat_dict = {conf_mat_dict}')
        print(f'\naccuracy = {accuracy}')

        # conf_mat = ConfusionMatrix(conf_mat_dict, m=1, M=50)
        # conf_mat.create_conf_matrix_save_to_html_file(f'conf_mat_{time.time()}.html')
        print(f'\n\n\navg accuracy for {_ + 1} runs = {np.mean(accuracies)}')

    print(f'\n\n\nAVG ACCURACY for {number_of_runs} runs = {np.mean(accuracies)}')


def main():
    start_1 = time.time()
    run_model_1()
    stop_1 = time.time()
    print(f'WHOLE MODEL 1 TOOK {stop_1 - start_1} SECS')

    start_2 = time.time()
    run_model_2()
    stop_2 = time.time()
    print(f'WHOLE MODEL 2 TOOK {stop_2 - start_2} SECS')


if __name__ == "__main__":
    main()
