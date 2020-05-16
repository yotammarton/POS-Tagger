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
import os


# TODO run generate_comp_tagged.py and see the results are the same as our comp tagged files - BE CAREFUL - OVERWRITES!!!!

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

        # Init all features classes dictionaries
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

        # Additional patterns
        self.class110_dict = OrderedDict()  # {(110.x, tag): # times seen}
        self.class111_dict = OrderedDict()  # {(111.x, tag): # times seen}

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
            Description of the sub-classes in our numbers class:

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
        Create counts dict for class 109 features - Capital Letters
        Description of the sub-classes for capital letters treatment:
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
        """
        Create counts dict for class 110 features
        TODO update documentation
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    cur_word = splited_words[word_idx].split('_')[0]
                    if len(cur_word) >= 13:
                        if re.search('ally$', cur_word) or re.search('ely$', cur_word) or \
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
                                re.search('tly$', cur_word):
                            if re.match('^[A-Z]$', cur_word[0]):  # NNP tag
                                if (110.135, cur_tag) not in self.class110_dict:
                                    self.class110_dict[(110.135, cur_tag)] = 1
                                else:
                                    self.class110_dict[(110.135, cur_tag)] += 1
                        elif not (re.search('[\-]', cur_word)):  # NN + NNS tags
                            if cur_word[-1] == 's' and not \
                                    re.search('ness$', cur_word) and not re.match('^[A-Z]$', cur_word[0]):
                                if (110.2, cur_tag) not in self.class110_dict:
                                    self.class110_dict[(110.2, cur_tag)] = 1
                                else:
                                    self.class110_dict[(110.2, cur_tag)] += 1
                    if re.match('^[0-9\-,.:]*[][0-9]+[0-9\-,.:]*$', cur_word) or (cur_word in ["II", "III", "IV"] or
                                                                                  re.match(
                                                                                      '^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$',
                                                                                      cur_word)):  # CD tag
                        if cur_word in ["II", "III", "IV"]:  # in big model its NNP and not CD
                            if (110.35, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.35, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.35, cur_tag)] += 1

                        elif (110.3, cur_tag) not in self.class110_dict:
                            self.class110_dict[(110.3, cur_tag)] = 1
                        else:
                            self.class110_dict[(110.3, cur_tag)] += 1

                    elif (re.search('[\-]', cur_word) and not re.match('^[A-Z]', cur_word.split('-')[-1])) and \
                            (re.search('ing$', cur_word.split('-')[-1]) or
                             re.search('ed$', cur_word.split('-')[-1]) or
                             re.search('ic$', cur_word.split('-')[-1]) or
                             re.search('age$', cur_word.split('-')[-1]) or
                             re.search('like$', cur_word.split('-')[-1]) or
                             re.search('ive$', cur_word.split('-')[-1]) or
                             re.search('ven$', cur_word.split('-')[-1]) or
                             re.search('^pre', cur_word.split('-')[0]) or
                             re.search('^anti', cur_word.split('-')[0]) or
                             re.search('er$', cur_word.split('-')[0])):  # JJ tag. also ~40 NN get in.
                        if (110.4, cur_tag) not in self.class110_dict:
                            self.class110_dict[(110.4, cur_tag)] = 1
                        else:
                            self.class110_dict[(110.4, cur_tag)] += 1

                    elif cur_word not in ["-LCB-", "-RCB-", "-LRB-", "-RRB-", "--", "...", "I", "A", ",", ".", ":"]:
                        if (re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', cur_word) and  # NNS tag
                            len(cur_word) != 2) and not \
                                re.match('^([a-zA-Z]*(-)?[0-9][0-9][0-9][0-9](s?))$|^([a-zA-Z]*(-)?[0-9][0-9]s)$',
                                         cur_word):
                            if (110.5, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.5, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.5, cur_tag)] += 1

                        if re.match('^[A-Z\-0-9.,]+$', cur_word) or \
                                re.search('[a-z\-][A-Z]', cur_word) or \
                                re.match('^[A-Za-z][\-][a-z]+$', cur_word):  # NNP tag
                            if (110.6, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.6, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.6, cur_tag)] += 1

                        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', cur_word):  # JJ tag
                            if (110.7, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.7, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.7, cur_tag)] += 1

                        if re.search('\.$', cur_word) and cur_word not in [".",
                                                                           "No."]:  # in big model its NNP, and not FW
                            if (110.9, cur_tag) not in self.class110_dict:
                                self.class110_dict[(110.9, cur_tag)] = 1
                            else:
                                self.class110_dict[(110.9, cur_tag)] += 1

    def set_class111_dict(self):
        """
        Create counts dict for class 111 features
        TODO update documentation
        """
        with open(self.file_path) as f:
            for line in f:
                splited_words = re.split(' |[\n]', line)
                if splited_words[-1] == "":
                    del splited_words[-1]  # remove \n
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx].split('_')[1]
                    cur_word = splited_words[word_idx].split('_')[0]
                    if len(cur_word) >= 13:
                        if re.search('ing$', cur_word) and not (re.search('[\-]', cur_word)):
                            if (111.11, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.11, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.11, cur_tag)] += 1

                        elif re.search('ed$', cur_word) and not (re.search('[\-]', cur_word)):
                            if (111.14, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.14, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.14, cur_tag)] += 1

                        elif re.search('ally$', cur_word) or re.search('ely$', cur_word) or \
                                re.search('tly$', cur_word):
                            if (111.12, cur_tag) not in self.class111_dict:  # RB tag
                                self.class111_dict[(111.12, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.12, cur_tag)] += 1

                        elif re.search('tant$', cur_word) or \
                                re.search('cal$', cur_word) or \
                                re.search('ic$', cur_word) or \
                                re.search('ive$', cur_word) or \
                                re.search('nal$', cur_word) or \
                                re.search('-dependent$', cur_word) or \
                                re.search('-sensitive$', cur_word) or \
                                re.search('-specific$', cur_word) or \
                                re.search('tly$', cur_word):  # JJ tag
                            if (111.13, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.13, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.13, cur_tag)] += 1
                        else:  # NN + NNS tags
                            if cur_word[-1] == 's':
                                if (111.2, cur_tag) not in self.class111_dict:
                                    self.class111_dict[(111.2, cur_tag)] = 1
                                else:
                                    self.class111_dict[(111.2, cur_tag)] += 1

                            if (111.1, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.1, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.1, cur_tag)] += 1

                    if re.match('^[0-9\-,.:]*[][0-9]+[0-9\-,.:]*$', cur_word) or (cur_word in ["II", "III", "IV"] or
                                                                                  re.match(
                                                                                      '^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$',
                                                                                      cur_word)):  # CD tag
                        if (111.3, cur_tag) not in self.class111_dict:
                            self.class111_dict[(111.3, cur_tag)] = 1
                        else:
                            self.class111_dict[(111.3, cur_tag)] += 1

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
                          re.search('kDa$', cur_word)):  # JJ tag
                        if (111.4, cur_tag) not in self.class111_dict:
                            self.class111_dict[(111.4, cur_tag)] = 1
                        else:
                            self.class111_dict[(111.4, cur_tag)] += 1

                    else:
                        if re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', cur_word):  # NNS tag
                            if (111.5, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.5, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.5, cur_tag)] += 1

                        if (re.match('^[A-Z\-0-9.,]+$', cur_word) and cur_word not in ["I", "A", ",", ".", ":"]) or \
                                re.search('[a-z\-][A-Z]', cur_word) or \
                                re.match('^[A-Za-z][\-][a-z]+$', cur_word) or \
                                re.match('^[a-z\-]+[0-9]+$', cur_word) or \
                                re.search('coid$', cur_word) or \
                                re.search('ness$', cur_word):  # NN tag
                            if (111.6, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.6, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.6, cur_tag)] += 1

                        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', cur_word):  # might be JJ tag
                            if (111.7, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.7, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.7, cur_tag)] += 1

                        if re.match('^[A-Z]?[a-z]+[\-][0-9\-.,]+$', cur_word):  # might be NN tag
                            if (111.8, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.8, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.8, cur_tag)] += 1

                        if re.search('\.$', cur_word) and cur_word != ".":  # might be FW tag, but not only
                            if (111.9, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.9, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.9, cur_tag)] += 1

                        if cur_word in ['Treponema', 'cerevisiae', 'pallidum', 'Borrelia', 'burgdorferi',
                                        'vitro', 'vivo', 'i.e.', 'e.g.']:  # FW tag
                            if (111.92, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.92, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.92, cur_tag)] += 1
                        if cur_word in ['in', 'In'] and word_idx != len(splited_words) - 1:  # FW tag
                            next_word = splited_words[word_idx + 1].split('_')[0]
                            if next_word in ['vitro', 'vivo']:
                                if (111.92, cur_tag) not in self.class111_dict:
                                    self.class111_dict[(111.92, cur_tag)] = 1
                                else:
                                    self.class111_dict[(111.92, cur_tag)] += 1

                        if cur_word in ['i', 'ii', 'iii', 'iv']:  # LS tag
                            if (111.93, cur_tag) not in self.class111_dict:
                                self.class111_dict[(111.93, cur_tag)] = 1
                            else:
                                self.class111_dict[(111.93, cur_tag)] += 1


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

        self.class111_feature_index_dict = OrderedDict()
        self.n_class111 = 0

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

    def set_index_class111(self, threshold=0):
        for key, value in self.feature_statistics.class111_dict.items():
            if value >= threshold:
                self.class111_feature_index_dict[key] = self.n_class111 + self.n_total_features
                self.n_class111 += 1
        self.n_total_features += self.n_class111

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
        self.all_feature_index_dict.update(self.class111_feature_index_dict)


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

    def plot_confusion_matrix(self, output_path: str = '', colored=False):
        """
        creates confusion matrix.
         if colored = False only plot the DataFrame to console
         but if colored = True create beautiful colored table and save to .html file with path 'outputh_path'
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
        :param output_path: the path for .html file to save (only saves if colored = True)
        :param colored: if True will save .html file with colored confusion matrix
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

        #  plot the regular DataFrame without colors
        if not colored:
            print(df)
            return

        # else: plot styled DataFrame with colors

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

        # write the pandas styler object to HTML -> can view the confusion matrix from any web browser
        with open(output_path, "w") as html:
            html.write('<font size="10" face="Courier New" >' + style.render() + '</font>')


# Auxiliary function for class f108
def re_match_words(regular_exp: str, lst):
    if not [w for w in lst if w != '']:
        return False
    for word in lst:
        if word != '' and not re.match(regular_exp, word):
            return False
    return True


# Auxiliary function for class f108
def re_match_letters_numbers(regular_exps: list, lst):
    for i, word in enumerate(lst):
        if i % 2 == 0:
            if word != '' and not re.match(regular_exps[i % 2], word):
                return False
        else:
            if word != '' and not re_match_words(regular_exps[i % 2], re.split('[,]|[:]|[\\\\]|[/]|[%]', word)):
                return False
    return True


# Auxiliary function for class f108
def re_match_numbers_letters(regular_exps: list, lst):
    for i, word in enumerate(lst):
        if i % 2 == 0:
            if word != '' and not re_match_words(regular_exps[i % 2], re.split('[,]|[:]|[\\\\]|[/]|[%]', word)):
                return False
        else:
            if word != '' and not re.match(regular_exps[i % 2], word):
                return False
    return True


"""START OF SECTION FOR OBJECTIVE FUNCTION AND GRADIENT CALC"""


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


def initialize_global_variables():
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


def function_l_and_gradient_l(v: np.array, *args):
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

        # finished collecting all data

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

    return -1 * (objective_value - (lam / 2) * (np.linalg.norm(v) ** 2)), -1 * (gradient_value - lam * v)


"""END OF SECTION OBJECTIVE FUNCTION AND GRADIENT CALC"""


def f_xi_yi(features_indices: Feature2Id, words, tags, i):
    """
    determine what features are fired for the given words and tags and location (history)
    :param features_indices: object containing the features and their indices
    :param words: words in sentence
    :param tags: tags
    :param i: location in sentence
    :return: list of all active features for the given history
    """
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
    elif (100, cur_word.lower(), cur_tag) in features_indices.class100_feature_index_dict:
        active_features_indices.append(features_indices.all_feature_index_dict[(100, cur_word.lower(), cur_tag)])
    elif (100, cur_word.upper(), cur_tag) in features_indices.class100_feature_index_dict:
        active_features_indices.append(features_indices.all_feature_index_dict[(100, cur_word.upper(), cur_tag)])
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
    if len(cur_word) >= 13:
        if re.search('ally$', cur_word) or re.search('ely$', cur_word) or re.search('tly$', cur_word):  # RB tag
            if (110.12, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.12, cur_tag)])

        elif re.search('tant$', cur_word) or \
                re.search('cal$', cur_word) or \
                re.search('ic$', cur_word) or \
                re.search('ive$', cur_word) or \
                re.search('nal$', cur_word) or \
                re.search('-dependent$', cur_word) or \
                re.search('-sensitive$', cur_word) or \
                re.search('-specific$', cur_word) or \
                re.search('tly$', cur_word):
            if re.match('^[A-Z]$', cur_word[0]):  # NNP tag
                if (110.135, cur_tag) in features_indices.class110_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(110.135, cur_tag)])

        elif not (re.search('[\-]', cur_word)):  # NN + NNS tags
            if cur_word[-1] == 's' and not \
                    re.search('ness$', cur_word) and not re.match('^[A-Z]$', cur_word[0]):
                if (110.2, cur_tag) in features_indices.class110_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(110.2, cur_tag)])

    if re.match('^[0-9\-,.:]*[0-9]+[0-9\-,.:]*$', cur_word) or cur_word in ["II", "III", "IV"] or \
            re.match('^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$', cur_word):  # CD tag
        if cur_word in ["II", "III", "IV"]:  # in big model its NNP and not CD
            if (110.35, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.35, cur_tag)])
        elif (110.3, cur_tag) in features_indices.class110_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(110.3, cur_tag)])

    elif (re.search('[\-]', cur_word) and not re.match('^[A-Z]', cur_word.split('-')[-1])) and \
            (re.search('ing$', cur_word.split('-')[-1]) or
             re.search('ed$', cur_word.split('-')[-1]) or
             re.search('ic$', cur_word.split('-')[-1]) or
             re.search('age$', cur_word.split('-')[-1]) or
             re.search('like$', cur_word.split('-')[-1]) or
             re.search('ive$', cur_word.split('-')[-1]) or
             re.search('ven$', cur_word.split('-')[-1]) or
             re.search('^pre', cur_word.split('-')[0]) or
             re.search('^anti', cur_word.split('-')[0]) or
             re.search('er$', cur_word.split('-')[0])):  # JJ tag
        if (110.4, cur_tag) in features_indices.class110_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(110.4, cur_tag)])
    elif cur_word not in ["-LCB-", "-RCB-", "-LRB-", "-RRB-", "--", "...", "I", "A", ",", ".", ":"]:
        if (re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', cur_word) and  # NNS tag
            len(cur_word) != 2) and not \
                re.match('^([a-zA-Z]*(-)?[0-9][0-9][0-9][0-9](s?))$|^([a-zA-Z]*(-)?[0-9][0-9]s)$', cur_word):
            if (110.5, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.5, cur_tag)])
        if re.match('^[A-Z\-0-9.,]+$', cur_word) or \
                re.search('[a-z\-][A-Z]', cur_word) or \
                re.match('^[A-Za-z][\-][a-z]+$', cur_word):  # NNP tag
            if (110.6, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.6, cur_tag)])

        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', cur_word):  # JJ tag
            if (110.7, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.7, cur_tag)])

        if re.search('\.$', cur_word) and cur_word not in [".", "No."]:  # in big model its NNP, and not FW
            if (110.9, cur_tag) in features_indices.class110_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(110.9, cur_tag)])

    # features fired in class 111
    if len(cur_word) >= 13:
        if re.search('ing$', cur_word) and not (re.search('[\-]', cur_word)):
            if (111.11, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.11, cur_tag)])
        elif re.search('ed$', cur_word) and not (re.search('[\-]', cur_word)):
            if (111.14, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.14, cur_tag)])

        elif re.search('ally$', cur_word) or re.search('ely$', cur_word) or re.search('tly$', cur_word):  # RB tag
            if (111.12, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.12, cur_tag)])

        elif re.search('tant$', cur_word) or \
                re.search('cal$', cur_word) or \
                re.search('ic$', cur_word) or \
                re.search('ive$', cur_word) or \
                re.search('nal$', cur_word) or \
                re.search('-dependent$', cur_word) or \
                re.search('-sensitive$', cur_word) or \
                re.search('-specific$', cur_word) or \
                re.search('tly$', cur_word):  # JJ tag
            if (111.13, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.13, cur_tag)])
        else:  # NN + NNS tags
            if cur_word[-1] == 's':
                if (111.2, cur_tag) in features_indices.class111_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(111.2, cur_tag)])

            if (111.1, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.1, cur_tag)])

    if re.match('^[0-9\-,.:]*[0-9]+[0-9\-,.:]*$', cur_word) or cur_word in ["II", "III", "IV"] or \
            re.match('^[0-9\-.]+[L][R][B][0-9\-.]+[R][R][B][0-9\-.]+$', cur_word):  # CD tag
        if (111.3, cur_tag) in features_indices.class111_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(111.3, cur_tag)])

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
          re.search('kDa$', cur_word)):  # JJ tag
        if (111.4, cur_tag) in features_indices.class111_feature_index_dict:
            active_features_indices.append(features_indices.all_feature_index_dict[(111.4, cur_tag)])
    else:
        if re.match('^[a-z]*[A-Z\-0-9.,]+[s]$', cur_word):  # NNS tag
            if (111.5, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.5, cur_tag)])

        if (re.match('^[A-Z\-0-9.,]+$', cur_word) and cur_word not in ["I", "A", ",", ".", ":"]) or \
                re.search('[a-z][A-Z]', cur_word) or \
                re.match('[A-Za-z][\-][A-Za-z0-9.,\-]+$', cur_word) or \
                re.search('coid$', cur_word) or re.search('ness$', cur_word):  # NN tag
            if (111.6, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.6, cur_tag)])

        if re.match('^[0-9\-.,]+[\-][a-zA-Z]+$', cur_word):  # might be JJ tag
            if (111.7, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.7, cur_tag)])

        if re.match('^[A-Z]?[a-z]+[\-][0-9\-.,]+$', cur_word):  # might be NN tag.
            if (111.8, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.8, cur_tag)])

        if re.search('\.$', cur_word) and cur_word != ".":  # might be FW tag, but maybe more
            if (111.9, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.9, cur_tag)])

        if cur_word in ['Treponema', 'pallidum', 'Borrelia', 'burgdorferi', 'vitro', 'vivo']:  # FW tag
            if (111.92, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.92, cur_tag)])

        if cur_word in ['in', 'In'] and i != len(words) - 1:  # FW tag
            next_word = words[i + 1]
            if next_word in ['vitro', 'vivo']:
                if (111.92, cur_tag) in features_indices.class111_feature_index_dict:
                    active_features_indices.append(features_indices.all_feature_index_dict[(111.92, cur_tag)])

        if cur_word in ['i', 'ii', 'iii', 'iv']:  # LS tag
            if (111.93, cur_tag) in features_indices.class111_feature_index_dict:
                active_features_indices.append(features_indices.all_feature_index_dict[(111.93, cur_tag)])

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
    """
    auxiliary function for calculating q parameters for Viterbi algorithm
    """
    # Notice that len(words) != len(tags) , but its ok
    return sum(weights[f_xi_yi(features_indices, words=words, tags=[t, u, v], i=2)])


def q_params_calc(k, t, u, weights, features_indices: Feature2Id, words: list, class_statistics: ClassStatistics):
    """
    calculate q parameters for Viterbi algorithm
    """
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
    :param class_statistics: relevant ClassStatistics object
    :param weights: chosen weights vector. will not be changed while the viterbi algorithm
    :param features_indices: relevant Feature2Id object
    :param words: a sentence to tag
    :param beam: number for beam search e.g: np.inf , 5, 10
    :return: inference of tags sequence
    """
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

    # Deterministic tagging
    for i in range(len(words)):
        if words[i] in [";", "--"]:
            tags_infer[i] = ":"

    return tags_infer


def inference(path_file_to_tag: str, path_result: str, weights, features_indices: Feature2Id,
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


def evaluate(path_true: str, path_predicted: str, class_statistics: ClassStatistics) -> (dict, float):
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

        tags_true = [w.split('_')[1] for w in splited_true]
        tags_predicted = [w.split('_')[1] for w in splited_predicted]
        for true, predicted in zip(tags_true, tags_predicted):
            if true != predicted:
                wrong += 1
            else:
                correct += 1

            if (true, predicted) in dict_for_confusion_matrix:
                dict_for_confusion_matrix[(true, predicted)] += 1
            else:  # for the case where we have some tag in the test that we didn't see during train
                dict_for_confusion_matrix[(true, predicted)] = 1

    return dict_for_confusion_matrix, float(correct / (correct + wrong))


def train_model_1(train_file_path_: str, factr_, lambda_):
    # in case of several runs in the same code
    initialize_global_variables()

    # create statistics object
    c = ClassStatistics(train_file_path_)
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

    # create features from statistics
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

    # build the final features dictionary after thresholds
    f.build_all_classes_feature_index_dict()

    # create initial weights vector
    x0 = np.random.randn(f.n_total_features)

    # run optimization
    # args= (file_path, lambda , feature_statistics, features_indices)
    args = (train_file_path_, lambda_, c, f)
    optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l, x0=x0, args=args, factr=factr_)

    # save .pkl file for statistics, features and weights objects (only for later use in generate_comp_tagged.py)
    c_path = rf'c_object_trained_on_{train_file_path_}.pkl'
    f_path = rf'f_object_trained_on_{train_file_path_}.pkl'
    weights_path = rf'weights_trained_on_{train_file_path_}.pkl'

    # extract weights
    weights = optimal_params[0]

    # write to .pkl
    with open(c_path, 'wb') as file:
        pickle.dump(c, file)
    with open(f_path, 'wb') as file:
        pickle.dump(f, file)
    with open(weights_path, 'wb') as file:
        pickle.dump(weights, file)

    return c, f, weights


def train_model_2(train_file_path_: str, lambda_):
    # in case of several runs in the same code
    initialize_global_variables()

    # create statistics object
    c = ClassStatistics(train_file_path_)
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
    c.set_class111_dict()

    # create features from statistics
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
    f.set_index_class111()

    # build the final features dictionary after thresholds
    f.build_all_classes_feature_index_dict()

    # create initial weights vector
    x0 = np.random.randn(f.n_total_features)

    # run optimization
    # args= (file_path, lambda , feature_statistics, features_indices)
    args = (train_file_path_, lambda_, c, f)
    optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l, x0=x0, args=args)

    # save .pkl file for statistics, features and weights objects (only for later use in generate_comp_tagged.py)
    c_path = rf'c_object_trained_on_{train_file_path_}.pkl'
    f_path = rf'f_object_trained_on_{train_file_path_}.pkl'
    weights_path = rf'weights_trained_on_{train_file_path_}.pkl'

    # extract weights
    weights = optimal_params[0]

    # write to .pkl
    with open(c_path, 'wb') as file:
        pickle.dump(c, file)
    with open(f_path, 'wb') as file:
        pickle.dump(f, file)
    with open(weights_path, 'wb') as file:
        pickle.dump(weights, file)

    return c, f, weights


def create_files_leave_one_out_model_2(train_file_path='train2.wtag'):
    """
    creates files splited to train and test for model 2, with test size = 1 (leave-one-out)
    :param train_file_path: path for original tagged train file
    :return: None, writes the files to directory: 'kfold_loo'
    """
    # prepare sentences list from tagged file
    with open(train_file_path) as file:
        sentences = [line for line in file]
    np.random.shuffle(sentences)  # shuffle the data before performing leave-one-out

    size = len(sentences)  # 250 for train2.wtag
    for i in range(size):
        test_indices = [i]
        train_indices = [idx for idx in range(size) if idx not in test_indices]
        # write test lines to file
        with open(os.path.join("kfold_loo", f"test2_{i + 1}.txt"), 'w') as file:
            for idx in test_indices:
                file.write(sentences[idx])

        # write train lines to file
        with open(os.path.join("kfold_loo", f"train2_{i + 1}.txt"), 'w') as file:
            for idx in train_indices:
                file.write(sentences[idx])


def run_leave_one_out_model_2():
    """
    call this function after 'create_files_leave_one_out_model_2' called
    evaluate leave-one-out accuracy for model 2
    we used this in order to get the most accurate prediction we can on model 2 performance
    """
    accuracies = list()
    for i in range(1, 251):
        initialize_global_variables()  # in case of several runs in the same code
        train_file_path = os.path.join("kfold_loo", f'train2_{i}.txt')
        test_file_path = os.path.join("kfold_loo", f'test2_{i}.txt')

        lambda_ = 0.02
        beam_ = 50
        result_file_path = f'{test_file_path.split(".")[0]}_tagged_{int(time.time())}.txt'

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
        c.set_class111_dict()

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
        f.set_index_class111()

        f.build_all_classes_feature_index_dict()

        args = (train_file_path, lambda_, c, f)
        x0 = np.random.randn(f.n_total_features)

        optimal_params = optimize.fmin_l_bfgs_b(func=function_l_and_gradient_l, x0=x0, args=args)
        v_star = optimal_params[0]

        inference(path_file_to_tag=test_file_path, path_result=result_file_path, weights=v_star,
                  features_indices=f, class_statistics=c, beam=beam_)

        conf_mat_dict, accuracy = evaluate(path_true=test_file_path, path_predicted=result_file_path,
                                           class_statistics=c)
        accuracies.append(accuracy)

        print(f'accuracy for run {i} = {accuracy}')
        print(f'avg accuracy for {i} runs = {np.mean(accuracies)}\n')

    print(f'ACCURACY FOR LEAVE-ONE-OUT MODEL 2 = {np.mean(accuracies)}')


def main():
    """
    train all models, inference on all files
    """

    """ ---------------------------------------------------------------------------------------------- """
    """ ########## TRAIN MODEL 1 ON train1.wtag AND INFERENCE ON: [train1.wtag, test1.wtag] ########## """

    # train model 1 on train1.wtag
    start = time.time()
    c, f, weights = train_model_1(train_file_path_=r'train1.wtag', factr_=1e11, lambda_=0.2)
    stop = time.time()
    print(f'MODEL 1 TRAINING ON FILE train1.wtag TOOK {stop - start} SECS')

    # inference on train1.wtag
    start = time.time()
    inference(path_file_to_tag=r'train1.wtag', path_result=r'train1_tagged.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 1 INFERENCE ON train1.wtag TOOK {stop - start} SECS')

    conf_mat_dict, accuracy = evaluate(path_true=r'train1.wtag', path_predicted=r'train1_tagged.wtag',
                                       class_statistics=c)
    print(f'ACCURACY FOR MODEL 1 TRAINED ON train1.wtag AND INFERENCE ON train1.wtag = {accuracy}')

    # inference on test1.wtag
    start = time.time()
    inference(path_file_to_tag=r'test1.wtag', path_result=r'test1_tagged.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 1 INFERENCE ON test1.wtag TOOK {stop - start} SECS')

    conf_mat_dict, accuracy = evaluate(path_true=r'test1.wtag', path_predicted=r'test1_tagged.wtag',
                                       class_statistics=c)
    print(f'ACCURACY FOR MODEL 1 TRAINED ON train1.wtag AND INFERENCE ON test1.wtag = {accuracy}')

    conf_mat = ConfusionMatrix(conf_mat_dict, m=1, M=50)
    conf_mat.plot_confusion_matrix()

    """ --------------------------------------------------------------------------------------------------------- """
    """ ########## TRAIN MODEL 1 ON train1test1.wtag AND INFERENCE ON: [train1test1.wtag, comp1.words] ########## """

    # train model 1 on train1test1.wtag ->> this file is train1.wtag + test1.wtag together
    start = time.time()
    c, f, weights = train_model_1(train_file_path_=r'train1test1.wtag', factr_=1e11, lambda_=0.2)
    stop = time.time()
    print(f'MODEL 1 TRAINING ON FILE train1test1.wtag TOOK {stop - start} SECS')

    # inference on train1test1.wtag
    start = time.time()
    inference(path_file_to_tag=r'train1test1.wtag', path_result=r'train1test1_tagged.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 1 INFERENCE ON train1test1.wtag TOOK {stop - start} SECS')

    conf_mat_dict, accuracy = evaluate(path_true=r'train1test1.wtag', path_predicted=r'train1test1_tagged.wtag',
                                       class_statistics=c)
    print(f'ACCURACY FOR MODEL 1 TRAINED ON train1test1.wtag AND INFERENCE ON train1test1.wtag = {accuracy}')

    # inference on comp1.words
    start = time.time()
    inference(path_file_to_tag=r'comp1.words', path_result=r'comp_m1_308044296.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 1 INFERENCE ON comp1.words TOOK {stop - start} SECS')

    """ ----------------------------------------------------------------------------------------------- """
    """ ########## TRAIN MODEL 2 ON train2.wtag AND INFERENCE ON: [train2.wtag, comp2.words] ########## """

    # train model 2 on train2.wtag
    start = time.time()
    c, f, weights = train_model_2(train_file_path_=r'train2.wtag', lambda_=0.02)
    stop = time.time()
    print(f'MODEL 2 TRAINING ON FILE train2.wtag TOOK {stop - start} SECS')

    # inference on train2.wtag
    start = time.time()
    inference(path_file_to_tag=r'train2.wtag', path_result=r'train2_tagged.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 2 INFERENCE ON train2.wtag TOOK {stop - start} SECS')

    conf_mat_dict, accuracy = evaluate(path_true=r'train2.wtag', path_predicted=r'train2_tagged.wtag',
                                       class_statistics=c)
    print(f'ACCURACY FOR MODEL 2 TRAINED ON train2.wtag AND INFERENCE ON train2.wtag = {accuracy}')

    # inference on comp2.words
    start = time.time()
    inference(path_file_to_tag=r'comp2.words', path_result=r'comp_m2_308044296.wtag', weights=weights,
              features_indices=f, class_statistics=c, beam=50)
    stop = time.time()
    print(f'MODEL 2 INFERENCE ON comp2.words TOOK {stop - start} SECS')


if __name__ == "__main__":
    # run_leave_one_out_model_2()
    main()
