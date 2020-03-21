import warnings

import sys
sys.path.append('../')

import os
import re
import nltk
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import textacy
import time
import pandas as pd
import gensim
import pickle
import math
import spacy
import model.aspect

from bratreader.repomodel import RepoModel
from sklearn.feature_extraction.stop_words import  ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordDependencyParser
from senticnet.senticnet import SenticNet
from nltk.corpus import wordnet

nlp = spacy.load('en')
nlp_sim = spacy.load('en_core_web_lg')

os.environ['CLASSPATH'] = "./stanford"
# add stanford-parser, stanford-parser-3.8.0-models
os.environ['STANFORD_PARSER'] = "./stanford"
os.environ['STANFORD_MODELS'] = "./stanford"
os.environ['STANFORD_MODEL'] = "./stanford"
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk1.8.0_151"

STOP_LIST = list(ENGLISH_STOP_WORDS) + \
            ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "<", ">", "?", "~", "`", "/", ",", ".", "{", "}", "[", "]", "|", "\\"] + \
            ["n't", "'s", "'m", "ca", "'re", "'ve", "=", "+", "'d", "'ll"] + \
            list(stopwords.words('english'))
REVIEW_DATA_PATH = "../data/reviews/reviews_AI_ML_100/"


class Aspect:

    def __init__(self, word, is_explicit=True, aspect_category=None):
        self.is_explicit = is_explicit
        self.word = word
        self.occ_list = {}
        self.doc_freq = 0
        self.aspect_categories = aspect_category
        self.wv = []
        self.is_wv = False
        self.feature_id = None
        self.rule_nums = []


    def update(self, sent, doc_id, line, start, end, aspect_category, rule_num, is_explict=True):

        if doc_id not in self.occ_list.keys():
            self.occ_list[doc_id] = []
            self.occ_list[doc_id].append([line, start, end, sent])
        else:
            self.occ_list[doc_id].append([line, start, end, sent])

        self.doc_freq = len(self.occ_list)
        self.is_explicit = is_explict

        if not aspect_category:
            return

        if not self.aspect_categories:
            self.aspect_categories = {}

        if aspect_category not in self.aspect_categories.keys():
            self.aspect_categories[aspect_category] = 1
        else:
            self.aspect_categories[aspect_category] += 1

        self.rule_nums.append(rule_num)

        return



class DataPreprocessor:

    def load_book_reviews(self):
        REVIEWS = "../data/tagged_reviews/reviews_ann1"
        r_basuru = RepoModel(REVIEWS)
        docs = {}

        for index in range(0, 100):
            doc_basuru = r_basuru.documents[str(index)]
            sentences = []
            for sent in doc_basuru.sentences:
                if not sent:
                    continue
                sent_words = []
                sent_start = 0
                sent_end = 0
                new_sent = True
                big_words = []
                for ww in sent.words:
                    big_words.append(ww.form)

                big_sent = ' '.join(big_words)
                small_sents = nltk.sent_tokenize(big_sent)
                for w in sent.words:
                    if new_sent:
                        sent_start = w.start
                        new_sent = False
                    sent_words.append(w.form)
                    s = w.form
                    t_sent = ' '.join(sent_words)
                    if s.endswith('.'):
                        sent_end = w.end
                        sent_m_end = sent_end - sent_start
                        sent_m_start = 0
                        sentence = ' '.join(sent_words)

                        sentences.append(((sent_m_start, sent_m_end), sentence))
                        sent_words = []
                        new_sent = True


            docs[index] = sentences

        return docs


class AspectExtractor:

    def __init__(self, review_folder=REVIEW_DATA_PATH, docs=None):
        #if not docs:
        #    self.docs = DataPreprocessor().load_reviews(review_folder)
        #else:
        self.docs = docs
        self.aspect_list = []


    def char_index(self, sentence, word_index, word):

        sentence1 = re.split('(\s)', sentence) # Parentheses keep split character
        #temp_sent = sentence.split()

        #tmp_sent = []
        #for t in sentence1:
        #    tt = list(t)
        #    if tt[len(tt) - 1] == '.':
        #        tt.remove('.')

        #    t = ''.join(tt)
        #    tmp_sent.append(t)

        #sentence1 = tmp_sent

        start_char = sentence.find(word)
        if start_char == -1:
            return
        end_char = start_char + len(word)


        #matches = [(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', sentence)]
        #b, c = zip(*matches)

        #ww = b[word_index-1]
        #ii = c[word_index-1]

        #start_index = ii[0]

        #if word_index != len(sentence.split()):
        #    end_index   = ii[1] + 1
        #else:
        #    end_index = ii[1]


        #for match in re.finditer(word, sentence):
        #    start_index = match.start()
        #    end_index = match.end()


        #len(''.join(sentence1[:word_index]))
        return start_char, end_char


    def is_word_in_iac_lexicon(self, word):

        words = word.split()
        count = 0
        for iac in self.iac_list:
            tokens = iac.split()
            for w in words:
                if  w in STOP_LIST:
                    continue

                if w in tokens:
                    #count += 1
                    return True
        #if count == len(words):
        #    return  True
        return False


    def is_not_valid_aspects(self, ptext, sent, node):
        '''
        filter followings
        Numbers, pronouns, aux verbs, who, what, whom,
        :return:
        '''

        noise_words = ['he', 'she', 'it', 'you', 'i', 'we', 'they', 'have', 'had', 'should', 'could', 'would', 'was',
                       'were', 'that','his','There', 'may' ,'might','is', 'are', 'been', 'be', 'who', 'all','whom', 'what', 'which', 'will', 'all', 'wtf', '1)want', 'men', '\'m']

        is_not_valid_aspect = False

        if ptext in noise_words:
            is_not_valid_aspect = True

        return  is_not_valid_aspect

    
    def find_aspects_using_additional_rule_1(self, node, nodes, sent, feature_list, rule_num):

        try:
            conjs = nodes['deps']['conj']
            if conjs:
                for conj in conjs:
                    conj_text = nodes[conj]['word']
                    #print(conj_text)
                    conj_node = nodes[conj]
                    rule_num = 9

                    start_index, end_index = self.char_index(sent, conj_node['address'], conj_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, conj_node,  start_index, end_index, sent, feature_list, rule_num )
                    print("ADDITIONAL RULE 1: ", " :H- ", node['word'], " : dir_node- ", conj_text, " : ", sent)
        except:
            pass
        return  feature_list


    def find_opinion_target_extraction_pattern_aspects(self, nodes, f_node, f_start_index, f_end_index, sent, feature_list, rule_num):

        f_node = self.noun_compound_rule_2(f_node, nodes, sent)
        f_start_index, f_end_index = self.char_index(sent, f_node['address'], f_node['word'])
        feature_list.append((f_node, f_node['word'], f_start_index, f_end_index, rule_num))

        return feature_list

    def pre_process_sent(self, sent):

        sent = sent.lower()
        sent = sent.replace('-', ' ')
        sent = sent.replace('$', ' ')

        # Remove punctuation characters
        tokens = nlp(sent)
        new_tokens = []
        for token in tokens:
            # if token.pos_ == 'PUNCT' or token.pos_ == 'SYM':
            #    continue
            new_tokens.append(token.text)

        sent = ' '.join(new_tokens)
        # sent = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sent)

        return sent

    def dependency_rule_based_features(self, true_aspect_list, docs=None):
        print("[INFO] Extract dependency rule based features ....")

        if not docs:
            docs = self.docs

        # update feature - doc_id, tecxt, line index, start char, end char
        doc_id_list = []

        for aspect in true_aspect_list:
            doc_ids = aspect.occ_list.keys()
            for doc_id, occs in aspect.occ_list.items():
                for occ in occs:
                    pair = (doc_id, occ[0])
                    if pair not in doc_id_list:
                        doc_id_list.append(pair)


        dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

        xx = 0

        for doc_id, doc in docs.items():
            #print("1111111")

            for l_index, sent_span in enumerate(doc):

                sent = sent_span[1]
                span = sent_span[0]

                pair = (doc_id, span)
                # Extract only the tagged sentences
                #if xx > 245:
                #    self.tagged_sent = True
                #else:
                #self.tagged_sent = True

                # TODO - check whether sentence does not contain any aspect corpus word

                sent = self.pre_process_sent(sent)
                #print(sent)
                tokens = nlp(sent)
                #xx += 1
               # for token in tokens:
               #     if token.text in self.aspect_corpus:
               #         self.tagged_sent = True
               #         print(xx, ':', token)
               #         break

                #if pair in doc_id_list:
                #    self.tagged_sent = True
                #else:
                #    if xx < 245:
                #        xx += 1
                #        print(xx)

                #if not self.tagged_sent:
                #    continue
                xx += 1
                #if xx > 50:
                #    break
                if xx % 100 == 0:
                    print(xx)

                #if xx == 100:
                #    return self.aspect_list
                try:
                    #sent = sent.strip().rstrip()
                    dep_graphs = list(dep_parser.raw_parse(sent))
                    #print(dep_graphs)

                    feature_list = self.traverse_to_extract_aspects(dep_graphs, sent)
                    for (node, ptext, start_char, end_char, rule_num) in feature_list:
                        if ptext in STOP_LIST  or self.is_not_valid_aspects(ptext, sent, node):
                            continue
                        self.update_feature(sent, doc_id, ptext, span, start_char, end_char, rule_num, False)

                    feature_list2 = self.traverse_to_extract_aspects_2(dep_graphs, sent)
                    for (node, ptext, start_char, end_char, rule_num) in feature_list2:
                        if ptext in STOP_LIST  or self.is_not_valid_aspects(ptext, sent, node):
                            continue
                        self.update_feature(sent, doc_id, ptext, span, start_char, end_char, rule_num, False)

                except Exception as e:
                    print(e)
                    pass
                    #sys.exit(1)

        return self.aspect_list


    def traverse_to_extract_aspects_2(self, dep_graphs, sent):
        feature_list = []

        for dep_graph in dep_graphs:
            nodes = dep_graph.nodes

            #TODO - IDENTIFY NON SUBJECT NODES
            is_subject_sent = False
            for key, node in nodes.items():

                try:
                    if node['rel'] == 'nsubj' or node['rel'] == 'nsubjpass' or \
                            node['rel'] == 'csubj' or node['rel'] == 'csubjpass':
                        is_subject_sent = True

                except:
                    pass
            if is_subject_sent:
                continue
            h_list = []
            h_list_2 = []
            # ONLY FOR ADJECTIVE AND ADVERBS
            for key, node in nodes.items():

                #if node['tag'] == 'JJ' or node['tag'] == 'RB':
                #    h_list.append(node)

                try:
                    if node['tag'] == 'JJ' or node['tag'] == 'JJR' or node['tag'] == 'JJS' or node['tag'] == 'RB' or \
                            node['tag'] == 'RBR' or node['tag'] == 'RBS':
                        h_list.append(node)
                except:
                    pass

                h_list_2.append(node)


            for h_node in h_list:
                feature_list1 = self.non_subject_noun_rules(h_node, nodes, sent)

                for feature in feature_list1:
                    feature_list.append(feature)

            # SEPARATE FOR ALL THE TOKENS
            #for key, node in nodes.items():
            # TODO - We need non subject sentences here

            for h_node in h_list_2:
                feature_list_1 = self.non_subject_noun_rule_2(h_node, nodes, sent)

                for feature in feature_list_1:
                    feature_list.append(feature)


        return feature_list


    def traverse_to_extract_aspects(self, dep_graphs, sent):

        feature_list = []

        for dep_graph in dep_graphs:
            nodes = dep_graph.nodes

            subjects = {}
            h_t_list = []

            try:
                for key, node in nodes.items():
                    try:
                        if node['rel'] == 'nsubj' or node['rel'] == 'nsubjpass' or \
                                node['rel'] == 'csubj' or node['rel'] == 'csubjpass':
                            subjects[node['address']] = node
                    except:
                        pass

            except:
                pass
                #print("parsing error ..")

            for key, node in nodes.items():
                deps = node['deps']
                if not deps:
                    continue
                try:
                    sub_indexes = deps['nsubj']
                    for sub_index in sub_indexes:
                        if sub_index in subjects.keys():
                            h_t_list.append((subjects[sub_index], node)) # h_node, t_node

                except:
                    pass
                    #print("parisng error")

                try:
                    sub_indexes = deps['nsubjpass']
                    for sub_index in sub_indexes:
                        if sub_index in subjects.keys():
                            h_t_list.append((subjects[sub_index], node)) # h_node, t_node

                except:
                    pass
                    #print("parsing error")


                try:
                    sub_indexes = deps['csubj']
                    for sub_index in sub_indexes:
                        if sub_index in subjects.keys():
                            h_t_list.append((subjects[sub_index], node)) # h_node, t_node

                except:
                    pass
                    #print("parsing error")

                try:
                    sub_indexes = deps['csubjpass']
                    for sub_index in sub_indexes:
                        if sub_index in subjects.keys():
                            h_t_list.append((subjects[sub_index], node)) # h_node, t_node
                except:
                    pass
                    #print("parsing error ..")


            # Run all the rules to extract aspects
            for h_t in h_t_list:
                h_node = h_t[0]
                t_node = h_t[1]

                feature_list1 = self.subject_noun_rule_1(h_node, t_node, nodes, sent)
                feature_list2 = self.subject_noun_rule_2(h_node, t_node, nodes, sent)
                #feature_list2 = []
                for feature in feature_list1:
                    feature_list.append(feature)


                for feature in feature_list2:
                    feature_list.append(feature)


        return feature_list

    def noun_compound_rule_2(self, t_node, nodes, sent):

        compound_node = t_node.copy()
        # print(t_node['word'])
        try:
            # print(t_node['deps'])
            # noun_comp_modis = t_node['deps']['nn']

            reg_1 = r'<NOUN><NOUN>+'
            regs = [reg_1]
            # for index_1 in noun_comp_modis:
            #    index_2 = t_node['address']
                # print(nodes[index_1]['word'])
            nlp_sent = nlp(sent)

            compound_word = compound_node['word']
            start_index = 0
            end_index = 0
            for reg in regs:
                #pt = textacy.extract.matches(nlp_sent, reg)
                pt = textacy.extract.pos_regex_matches(nlp_sent, reg)
                for p in pt:
                    print(p)
                    if compound_word in p.text and len(compound_word) < len(p.text):
                        compound_word = p.text
                        start_index = p.start_char
                        end_index = p.end_char
                # if index_1 > index_2:
                #    for i in range(index_2, index_1+1):
                #        compound_word = " ".join([compound_word, nodes[i]['word']])
                # elif index_2 > index_1:
                #    for i in range(index_1, index_2+1):
                #        compound_word = " ".join([compound_word, nodes[i]['word']])
                # else:
                #    compound_word = t_node['word']
            #if compound_word != compound_node['word']:
            #    print(compound_word)
            compound_node['word'] = compound_word

        except Exception as e:
            #print(e)
            #sys.exit(1)
            pass

        return compound_node

    def find_t_t1(self, t_node, t1_node, nodes, sent):


        index_t = t_node['address']
        index_t1 = t1_node['address']
        compound_node = t_node.copy()

        compound_word = None
        if index_t > index_t1:
            for i in range(index_t1, index_t+1):
                if not compound_word:
                    compound_word = nodes[i]['word']
                else:
                    compound_word = " ".join([compound_word, nodes[i]['word']])

        elif index_t1 > index_t:
            for i in range(index_t, index_t1+1):
                if not compound_word:
                    compound_word = nodes[i]['word']
                else:
                    compound_word = " ".join([compound_word, nodes[i]['word']])
        else:
            compound_word = t_node['word']

        #print("t-t1: ", compound_word)
        compound_node['word'] = compound_word

        return compound_node


    def subject_noun_rule_1(self, h_node, t_node, nodes, sent):
        sn = SenticNet()
        feature_list = []
        rule_num = 1

        try:
            adv_mods = t_node['deps']['advmod']
            if adv_mods:
                for adv_mod in adv_mods:
                    adv_text = nodes[adv_mod]['word']
                    adv_node = nodes[adv_mod]
                    try:
                        concept_info = sn.concept(adv_text)
                        polarity = sn.polarity_value(adv_text)
                        #print(adv_text, ":", polarity)
                        intens = sn.polarity_intense(adv_text)
                        intense = float(intens)

                        #print("intense: ", intense)
                        #if intense <= -0.6 or intense >= 0.6:
                        if True:
                            t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                            start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num  )
                            feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)
                            #print("RULE 1-ADV: ", t_node['word'], " : ", adv_node['word'], sent)


                    except:
                        pass
        except:
            pass
            #print("parsing error.. ")

        # Handling adjective modifiers
        try:
            adj_mods = t_node['deps']['amod']
            if adj_mods:
                for adj_mod in adj_mods:
                    adj_text = nodes[adj_mod]['word']
                    adj_node = nodes[adj_mod]
                    try:
                        concept_info = sn.concept(adj_text)
                        intens = sn.polarity_intense(adj_text)
                        intense = float(intens)

                        if True:
                        #print("intense: ", intense)
                        #if intense <= -0.6 or intense >= 0.6:
                            t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                            start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                            feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)
                            #print("RULE 1-ADJ: ", t_node['word'], " : ", adj_node['word'], sent)
                    except:
                        pass
        except:
            pass
            #print("parsing error.. ")


        # NEW RULE - xcomp 1.2

        try:
            adj_mods = t_node['deps']['xcomp']
            if adj_mods:
                for adj_mod in adj_mods:
                    adj_text = nodes[adj_mod]['word']
                    adj_node = nodes[adj_mod]
                    try:
                        concept_info = sn.concept(adj_text)
                        intens = sn.polarity_intense(adj_text)
                        intense = float(intens)

                        if True:
                        #print("intense: ", intense)
                        #if intense <= -0.6 or intense >= 0.6:
                            rule_num = 11
                            t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                            start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                            feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)
                            #print("RULE 1-XCOMP: ", t_node['word'], " : ", adj_node['word'], sent)
                    except:
                        pass
        except:
            pass
            #print("parsing error...")

        return feature_list

    def subject_noun_rule_2(self, h_node, t_node, nodes, sent):


        # TODO - BETTER WAY TO FIND AUX VERBS
        is_aux_verb = False
        try :
            for key, node in nodes.items():
                if (node['tag'] == 'VB' or node['tag'] == 'VBD' or node['tag'] == 'VBG' or node['tag'] == 'VBN' or
                node['tag'] == 'VBP' or node['tag'] == 'VBZ')  and node['rel'] == 'aux':
                    is_aux_verb = True
                    break
        except:
            pass
            #print("parsing error ..")

        # Depend on the relation of t_node
        deps = None
        try:
            deps = t_node['deps']
        except:
            pass
            #print("deps errro ..")
        if not deps:
            return

        feature_list = []

        # START - SUBJECT NOUN RULE 2 ------------------------------------
        is_aux_verb = False
        if not is_aux_verb:
            # TODO:2.1. If the verb t is modified by an adjective or an adverb or it is in adverbial clause modifier
            # relation with another token, then both h and t are extracted as aspects
            try:
                adv_mods = deps['advmod']
                adv_text = ''

                for adj_mod in adv_mods:
                    adv_text = nodes[adj_mod]['word']

                    try:
                        if 'VB' not in t_node['ctag']:
                            continue
                    except:
                        pass
                #if adv_mods:
                    rule_num = 2.1
                    t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)

                    h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)

                    #print("RULE 2-ADV: ", adv_text," :T- ", t_node['word'], " : H- ", h_node['word'], " : ", sent)
            except:
                pass
                #print("parsing error")


            try:
                adj_mods = deps['amod']
                adj_text = ''
                for adj_mod in adj_mods:
                    adj_text = nodes[adj_mod]['word']
                #if adj_mods:
                    try:
                        if 'VB' not in t_node['ctag']:
                            continue
                    except:
                        pass
                    rule_num = 2.1

                    t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)

                    h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)

                    #print("RULE 2-ADJ: ", adj_text ," :T- ", t_node['word'], " : H- ", h_node['word'], " : ",sent)

            except:
                pass
                #print("parsing error")


            try:
                adv_clse_mods = deps['advcl']
                adv_clse = ''
                for adj_mod in adv_clse_mods:
                    adv_clse = nodes[adj_mod]['word']
                #if adv_clse_mods:

                    try:
                        if 'VB' not in t_node['ctag']:
                            continue
                    except:
                        pass

                    rule_num = 2.1
                    t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)

                    h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)

                    #print("RULE 2-ADJ_CLSE: ", adv_clse, " :T- ", t_node['word'], " : H- ", h_node['word'], " : ", sent)
            except:
                pass
                #print("parsing error --")


            try:
                # NEW RULE = xcomp 2.1
                adv_clse_mods = deps['xcomp']
                adv_clse = ''
                for adj_mod in adv_clse_mods:
                    adv_clse = nodes[adj_mod]['word']
                #if adv_clse_mods:
                    try:
                        if 'VB' not in t_node['ctag']:
                            continue
                    except:
                        pass

                    rule_num = 13
                    t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)

                    h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                    #print("NEW RULE : RULE 2-XCOMP: ", adv_clse ," :T- ", t_node['word'], " : H- ", h_node['word'], " : ",sent)
            except:
                pass
                #print("parsing error..")

            # TODO: 2.2. t has direct object relation with a token n and the POS of the token is Noun and n is not
            # in SenticNet, then n extracted as an aspect (NN, NNS, )
            # 2.3. direct object noun, noun not exist in SenticNet, another token n1 is connected to n using any dependancy and
            # n1 is a noun

            try:
                direct_objects = deps['dobj']
                if direct_objects:
                    for dir_obj_id in direct_objects:
                        dir_obj = nodes[dir_obj_id]
                        if dir_obj['ctag'] == 'NN' or dir_obj['ctag'] == 'NNS':

                            sn = SenticNet()
                            try:
                                rule_num = 2.3
                                concept_info = sn.concept(dir_obj['word'])
                                intense = sn.polarity_intense(dir_obj['word'])
                                if True:
                                #if intense <= -0.6 or intense >= 0.6:
                                    for key, dep_indexes in dir_obj['deps'].items():
                                        for index in dep_indexes:
                                            node_n1 = nodes[index]
                                            if dir_obj == node_n1:
                                                continue
                                            if node_n1['ctag'] == 'NN' or node_n1['ctag'] == 'NNS':
                                                node_n1 = self.noun_compound_rule_2(node_n1, nodes, sent)
                                                start_index, end_index = self.char_index(sent, node_n1['address'], node_n1['word'])
                                                feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_n1,start_index, end_index, sent, feature_list, rule_num )
                                                feature_list = self.find_aspects_using_additional_rule_1(node_n1, nodes, sent, feature_list, rule_num)
                                                #print("RULE 2.3_n1: ", " :N- ", dir_obj['word'], " : N1- ", node_n1['word'], " : ", sent)

                                    dir_obj = self.noun_compound_rule_2(dir_obj, nodes, sent)
                                    start_index, end_index = self.char_index(sent, dir_obj['address'], dir_obj['word'])
                                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, dir_obj,start_index, end_index, sent, feature_list, rule_num )
                                    feature_list = self.find_aspects_using_additional_rule_1(dir_obj, nodes, sent, feature_list, rule_num)
                                    #print("RULE 2.3_n: ", " :T- ", t_node['word'], " : DIR_OBJ- ", dir_obj['word'], " : ", sent)
                            except: #2.2
                                rule_num = 2.2
                                dir_obj = self.noun_compound_rule_2(dir_obj, nodes, sent)
                                start_index, end_index = self.char_index(sent, dir_obj['address'], dir_obj['word'])
                                feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, dir_obj,start_index, end_index, sent, feature_list, rule_num )# TODO - Find other aspects in sent
                                feature_list = self.find_aspects_using_additional_rule_1(dir_obj, nodes, sent, feature_list, rule_num)
                                #print("RULE 2.2: ", " :T- ", t_node['word'], " : DIR_OBJ- ", dir_obj['word'], " : ", sent)
            except:
                pass
                #print("parsing error ")

            try:
                # 2.4 t in in open clausal complement relation with a token t1,#
                casual_complements = deps['xcomp']
                if casual_complements:
                    for cc_id in casual_complements:
                        node_t1 = nodes[cc_id]
                        node_t_t1 = self.find_t_t1(t_node, node_t1, nodes, sent)

                        # print(t_node['word'], " : ", node_t1['word'])
                        #if True:
                        if self.is_word_in_iac_lexicon(t_node['word']) and self.is_word_in_iac_lexicon(node_t1['word']):

                            #print("SUCCESSFUL")
                            rule_num = 2.4
                            node_t_t1 = self.noun_compound_rule_2(node_t_t1, nodes, sent)
                            start_index, end_index = self.char_index(sent, node_t_t1['address'], node_t_t1['word'])
                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_t_t1,start_index, end_index, sent, feature_list, rule_num )
                            feature_list = self.find_aspects_using_additional_rule_1(node_t_t1, nodes, sent, feature_list, rule_num)
                            #print("RULE 2.4: ", " :T- ", t_node['word'], " : T1- ", node_t1['word'], " : ",  node_t_t1['word'], " : ", sent)


                            t1_deps = node_t1['deps']
                            for key, dep_indexes in t1_deps.items():
                                for index in dep_indexes:
                                    node_t2 = nodes[index]
                                    if node_t2 == node_t1:
                                        continue
                                    if node_t2['ctag'] == 'NN' or node_t2['ctag'] == 'NNS':
                                        node_t2 = self.noun_compound_rule_2(node_t2, nodes, sent)
                                        start_index, end_index = self.char_index(sent, node_t2['address'], node_t2['word'])
                                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_t2,start_index, end_index, sent, feature_list, rule_num )
                                        feature_list = self.find_aspects_using_additional_rule_1(node_t2, nodes, sent, feature_list, rule_num)
                                        #print("RULE 2.4: ", " :T2- ", node_t2['word'], " : T1- ", node_t1['word'], " : ",  node_t_t1['word'], " : ", sent)

            except:
                pass
                #print("parser error ..")
            # NEW RULE - casual component relation  2.4

            try:
                casual_complements = deps['ccomp']
                if casual_complements:
                    for cc_id in casual_complements:
                        node_t1 = nodes[cc_id]
                        node_t_t1 = self.find_t_t1(t_node, node_t1, nodes, sent)

                        # print(t_node['word'], " : ", node_t1['word'])
                        #if True:
                        if self.is_word_in_iac_lexicon(t_node['word']) and self.is_word_in_iac_lexicon(node_t1['word']):

                            #print("SUCCESSFUL")
                            rule_num = 12
                            node_t_t1 = self.noun_compound_rule_2(node_t_t1, nodes, sent)
                            start_index, end_index = self.char_index(sent, node_t_t1['address'], node_t_t1['word'])
                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_t_t1,start_index, end_index, sent, feature_list, rule_num )
                            feature_list = self.find_aspects_using_additional_rule_1(node_t_t1, nodes, sent, feature_list, rule_num)
                            #print("NEW RULE : RULE 2.4: ", " :T- ", t_node['word'], " : T1- ", node_t1['word'], " : ",  node_t_t1['word'], " : ", sent)


                            t1_deps = node_t1['deps']
                            for key, dep_indexes in t1_deps.items():
                                for index in dep_indexes:
                                    node_t2 = nodes[index]
                                    if node_t2 == node_t1:
                                        continue
                                    if node_t2['ctag'] == 'NN' or node_t2['ctag'] == 'NNS':
                                        node_t2 = self.noun_compound_rule_2(node_t2, nodes, sent)
                                        start_index, end_index = self.char_index(sent, node_t2['address'], node_t2['word'])
                                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_t2,start_index, end_index, sent, feature_list, rule_num )
                                        feature_list = self.find_aspects_using_additional_rule_1(node_t2, nodes, sent, feature_list, rule_num)
                                        #print("NEW RULE - RULE 2.4: ", " :T2- ", node_t2['word'], " : T1- ", node_t1['word'], " : ",  node_t_t1['word'], " : ", sent)
            except:
                pass
                #print("parsing error ")

        # END - SUBJECT NOUN RULE 2 ------------------------------------------------------------------------------------

        # 3., 4. Copula relation
        try:
            copulas = deps['cop']
            if copulas:

                # Rule 3 - copular verb should be in implicit aspect
                # Check whether copular verb in senticNet
                is_in_lexicon = False
                cc_verb = ''
                for copula_id in copulas:
                    copula_verb = nodes[copula_id]['word']
                    #print(copula_verb)
                    cc_verb = copula_verb

                    try:
                        #print("COPULA VERB: ", copula_verb, " : ", t_node['word'] )
                        if self.is_word_in_iac_lexicon(copula_verb):
                        #if copula_verb in self.iac_list:
                            is_in_lexicon = True
                    except:
                        pass

                #if True:
                if is_in_lexicon:
                    rule_num = 3
                    t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)
                    #print("RULE 3: ", " :T- ", t_node['word'], " : cc-verb- ", cc_verb, " : ", sent)

                # 4. if token t is in copula relation with copular verb and POS of h is noun
                # then h is extracted as
                is_h_noun = False
                try:
                    if h_node['ctag'] == 'NN' or h_node['ctag'] == 'NNS':
                        is_h_noun = True
                except:
                    pass
                    #print("parsing error")


                if is_h_noun:
                    #print("COPULA NOUN: ", h_node['word'])
                    rule_num = 4
                    h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                    #print("RULE 4: ", " :T- ", t_node['word'], " : cc-verb- ", cc_verb, " : ", "H: " , h_node['word'],  " : ", sent)


            #5.  t in copula relation with a copular verb and copula verb connected to t1 and t1 is a verb
            if copulas:

                is_t_in_iac_lexicon = False # TODO - Testing the importance
                t_word = t_node['word']

                try:
                    #concept = sn.concept(t_word)
                    #if t_word in self.iac_list:
                    if self.is_word_in_iac_lexicon(t_word):
                        is_t_in_iac_lexicon = True
                except:
                    pass

                #if True:
                if is_t_in_iac_lexicon:
                    for copula_id in copulas:
                        copula_node = nodes[copula_id]

                        try:
                            c_deps = t_node['deps']
                            #c_deps = t_node['deps']
                            for key, dp_indexes in c_deps.items():
                                for index in dp_indexes:
                                    node_t1 = nodes[index]
                                    if copula_node == node_t1:
                                        continue

                                    if node_t1['ctag'] == 'VB' or node_t1['ctag'] == 'VBD' or node_t1['ctag'] == 'VBN' \
                                        or node_t1['ctag'] == 'VBZ' or  node_t1['ctag'] == 'VBP' or node_t1['ctag'] == 'VBG':

                                        is_t1_in_iac_lexicon = False
                                        try:
                                            #concept = sn.concept(node_t1['word'])
                                            #if node_t1['word'] in self.iac_list:
                                            if self.is_word_in_iac_lexicon(node_t1['word']):
                                                is_t1_in_iac_lexicon = True
                                        except:
                                            pass

                                        #if True:
                                        if is_t1_in_iac_lexicon:
                                            rule_num = 5
                                            t_node = self.noun_compound_rule_2(t_node, nodes, sent)
                                            start_index, end_index = self.char_index(sent, t_node['address'], t_node['word'])
                                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t_node,start_index, end_index, sent, feature_list, rule_num )
                                            feature_list = self.find_aspects_using_additional_rule_1(t_node, nodes, sent, feature_list, rule_num)

                                            #print("RULE_5: ", t_node['word'], " : ", sent)
                                            node_t1 = self.noun_compound_rule_2(node_t1, nodes, sent)
                                            start_index, end_index = self.char_index(sent, node_t1['address'], node_t1['word'])
                                            feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, node_t1,start_index, end_index, sent, feature_list, rule_num )
                                            feature_list = self.find_aspects_using_additional_rule_1(node_t1, nodes, sent, feature_list, rule_num)

                                            #print("RULE_5_1: ", node_t1['word'], " : ", sent)
                        except:
                            pass
                            #print("parse error")

        except:
            pass
            #print("parsing error..")
                                    #print("RULE 5: ", " :T- ", t_node['word'], " : T1:  ", node_t1['word'], " : ", sent)
        return feature_list



    def non_subject_noun_rules(self, h_node, nodes, sent):

        feature_list = []

        try:

            h_deps = h_node['deps']
            cc_realtions_2 = h_deps['xcomp']
            #dep_relation = h_deps['dep']


            if cc_realtions_2 :
                # Rule 1: if h is in infinitival or open clausal complement relation with t and h exisits in implicit aspect
                # lexicon extract h as a aspcet

                # Check whether h is in SenticNet
                is_in_iac_lexicon = False

                for cc_id in cc_realtions_2:
                    cc_node = nodes[cc_id]
                    try:
                        # concept = sn.concept(h_node['word'])
                        #if h_node['word'] in self.iac_list:
                        if self.is_word_in_iac_lexicon(cc_node['word']) and self.is_word_in_iac_lexicon(h_node['word']):
                            is_in_iac_lexicon = True
                    except:
                        pass

                    #h_node = self.additional_compound_rule_2(h_node, nodes, sent)
                    #if True:
                    rule_num = 6
                    if is_in_iac_lexicon:
                        h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                        #print("NON RULE 1: ", " :H- ", h_node['word'], " : cc_node- ", cc_node['word'], " : ", sent)

        except:
            pass
            #print("parse error..")

        try:

            h_deps = h_node['deps']
            inf_relations = h_deps['infmod']
            if inf_relations:
                # Rule 1: if h is in infinitival or open clausal complement relation with t and h exisits in implicit aspect
                # lexicon extract h as a aspcet

                # Check whether h is in SenticNet
                is_in_iac_lexicon = False
                for inf_id in inf_relations:
                    inf_node = nodes[inf_id]
                    try:
                        # concept = sn.concept(h_node['word'])
                        #if h_node['word'] in self.iac_list:
                        if self.is_word_in_iac_lexicon(inf_node['word']) and self.is_word_in_iac_lexicon(h_node['word']):
                            is_in_iac_lexicon = True
                    except:
                        pass

                    #h_node = self.additional_compound_rule_2(h_node, nodes, sent)
                    #if True:
                    if is_in_iac_lexicon:
                        rule_num = 6
                        h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                        #print("NON RULE 1: ", " :H- ", h_node['word'], " : inf_node- ", inf_node['word'], " : ", sent)

        except:
            pass
            #print("parsing error.. ")


        #NEW RULE - cc_relations = h_deps['ccomp']
        try:

            h_deps = h_node['deps']
            inf_relations = h_deps['ccomp']
            if inf_relations:
                # Rule 1: if h is in infinitival or open clausal complement relation with t and h exisits in implicit aspect
                # lexicon extract h as a aspcet

                # Check whether h is in SenticNet
                is_in_iac_lexicon = False
                for inf_id in inf_relations:
                    inf_node = nodes[inf_id]
                    try:
                        # concept = sn.concept(h_node['word'])
                        #if h_node['word'] in self.iac_list:
                        if self.is_word_in_iac_lexicon(inf_node['word']) and self.is_word_in_iac_lexicon(h_node['word']):
                            is_in_iac_lexicon = True
                    except:
                        pass

                    #h_node = self.additional_compound_rule_2(h_node, nodes, sent)
                    #if True:
                    if is_in_iac_lexicon:
                        rule_num = 14
                        h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                        #print("NEW RULE - ccmp NON RULE 1: ", " :H- ", h_node['word'], " : inf_node- ", inf_node['word'], " : ", sent)

        except:
            pass


        #NEW RULE - cc_relations = h_deps['ccomp']
        try:

            h_deps = h_node['deps']
            inf_relations = h_deps['dep']
            if inf_relations:
                # Rule 1: if h is in infinitival or open clausal complement relation with t and h exisits in implicit aspect
                # lexicon extract h as a aspcet

                # Check whether h is in SenticNet
                is_in_iac_lexicon = False
                for inf_id in inf_relations:
                    inf_node = nodes[inf_id]
                    try:
                        # concept = sn.concept(h_node['word'])
                        #if h_node['word'] in self.iac_list:
                        if self.is_word_in_iac_lexicon(inf_node['word']) and self.is_word_in_iac_lexicon(h_node['word']):
                            is_in_iac_lexicon = True
                    except:
                        pass

                    #h_node = self.additional_compound_rule_2(h_node, nodes, sent)
                    #if True:
                    if is_in_iac_lexicon:
                        rule_num = 14
                        h_node = self.noun_compound_rule_2(h_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, h_node['address'], h_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, h_node,start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(h_node, nodes, sent, feature_list, rule_num)
                        #print("NEW RULE - dep NON RULE 1: ", " :H- ", h_node['word'], " : inf_node- ", inf_node['word'], " : ", sent)

        except:
            pass


        return feature_list

    def non_subject_noun_rule_2(self, h_node, nodes, sent):

        feature_list = []
        h_deps = h_node['deps']

        #print("NON SUBJECT NOUN RULE----------------")

        # RULE 3
        try:
            dir_objs  = h_deps['dobj']
            if dir_objs:
                for dir_id in dir_objs:

                    rule_num = 8
                    t1_node = nodes[dir_id]
                    t1_node = self.noun_compound_rule_2(t1_node, nodes, sent)
                    start_index, end_index = self.char_index(sent, t1_node['address'], t1_node['word'])
                    feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, t1_node,start_index, end_index, sent, feature_list, rule_num )
                    feature_list = self.find_aspects_using_additional_rule_1(t1_node, nodes, sent, feature_list, rule_num)
                    #print("NON RULE 3: ", " :H- ", h_node['word'], " : dir_node- ", t1_node['word'], " : ", sent)
        except:
            pass
            #print("parsing error ...")

        # RULE 2

        try:
            pre_mod1 = h_deps['prep']

            if pre_mod1:
                #print("PRE MODE 1 -------------")
                for pre_id in pre_mod1:
                    pre_node = nodes[pre_id]
                    #print("RULE 3: ", pre_node['word'])

                    if pre_node['ctag'] == 'NN' or pre_node['ctag'] == 'NNS':
                        rule_num = 7
                        pre_node = self.noun_compound_rule_2(pre_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, pre_node['address'], pre_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, pre_node, start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(pre_node, nodes, sent, feature_list, rule_num)
                        #print("NON RULE 2: ", " :H- ", h_node['word'], " : pre_node- ", pre_node['word'], " : ", sent)

        except:
            pass
            #print("parsing error..")

        try:
            pre_mod2 = h_deps['prepc']
            if pre_mod2:
                #print("PRE MOD 2 -----------------")
                for pre_id in pre_mod2:
                    pre_node = nodes[pre_id]
                    #print("RULE 3.1: ", pre_node['word'])

                    if pre_node['ctag'] == 'NN' or pre_node['ctag'] == 'NNS':
                        rule_num = 7
                        pre_node = self.noun_compound_rule_2(pre_node, nodes, sent)
                        start_index, end_index = self.char_index(sent, pre_node['address'], pre_node['word'])
                        feature_list = self.find_opinion_target_extraction_pattern_aspects(nodes, pre_node, start_index, end_index, sent, feature_list, rule_num )
                        feature_list = self.find_aspects_using_additional_rule_1(pre_node, nodes, sent, feature_list, rule_num)
                        #print("NON RULE 2: ", " :H- ", h_node['word'], " : inf_node- ", pre_node['word'], " : ", sent)

        except:
            pass
            #print("parsing error.. ")

        return feature_list


    def update_feature(self, sent, doc_id, word, line, start, end, rule_num,  is_explicit=True, aspect_category=None):

        is_feature = False
        for feature in self.aspect_list:
            if word == feature.word: #and is_explicit == feature.is_explicit: #TODOG - check impl, explicit
                feature.update(sent, doc_id, line, start, end, aspect_category, rule_num)
                is_feature = True
                #print(aspect_category)
                ass = self.aspect_list
                x = 0
                break

        if not is_feature:
            aspect = model.aspect.Aspect(word, is_explicit)
            aspect.update(sent, doc_id, line, start, end, aspect_category, rule_num)
            self.aspect_list.append(aspect)
            ass = self.aspect_list
            x = 0

        return

    def aspect_extraction_performance(self, pred_aspects, true_aspects, reviews):
        if not pred_aspects:
            print("[ERROR] Predicted aspects are empty ..")
            return

        if not true_aspects:
            print("[ERROR] true aspects are empty ..")
            return

        # DEBUG - find duplicates of predicted aspcest
        dupp = {}
        for asp in pred_aspects:
            key = asp.word
            if key in dupp.keys():
                dupp[key] += 1
            else:
                dupp[key] = 1

        # REMOVE DUPLICATES FROM PREDICTED ASPECTS
        for aspect in pred_aspects:
            occ_list = aspect.occ_list
            occ_dir = {}
            for key, occs in occ_list.items():
                occ_temp = []
                for occ in occs:
                    if occ not in occ_temp:
                        occ_temp.append(occ)
                occ_dir[key] = occ_temp
            aspect.occ_list = occ_dir


        # ADD THE NEWLY EXTRACTED ASPECTS
        # REMOVE DUPLICATES FROM
        for aspect in true_aspects:
            occ_list = aspect.occ_list
            occ_dir = {}
            occ_list_tmp = []
            for key, occs in occ_list.items():
                occ_temp = []
                for occ in occs:
                    if occ not in occ_temp:
                        occ_temp.append(occ)
                        occ_list_tmp.append(occ)

                occ_dir[key] = occ_temp
            aspect.occ_list = occ_dir

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        fp_words = []
        ii = 0
        print("true apsects again: ", len(true_aspects))
        final_results = {}
        self.final_aspects_clusters = []
        self.fp_aspects = []

        #aspect_rules = {}
        # todo - CREATE DATA CSV FILE <>


        for true_aspect in true_aspects:
            true_occ_list = true_aspect.occ_list
            for doc_id, occs_true in true_occ_list.items():

                occs_pred = []
                for pred_aspect in pred_aspects:
                    pred_occ_list = pred_aspect.occ_list
                    for doc_id_pred, occs_pred_temp in pred_occ_list.items():
                        if doc_id == doc_id_pred:
                            for occ in occs_pred_temp:
                                occs_pred.append((occ, pred_aspect.word, pred_aspect))
                if len(occs_pred) == 0:
                    continue

                pred_sent_spans = []
                for op, word, ass in occs_pred:
                    span = op[0]
                    pred_sent_spans.append(span)

                for occ_true in occs_true:

                    is_fn = True
                    is_tp = False
                    sent_found = False
                    sentence = occ_true[3]

                    true_sent_span = occ_true[0]
                    if true_sent_span not in pred_sent_spans:
                        fn += 1

                        # TODO - DUMPING DATA
                        if doc_id not in final_results.keys():
                            final_results[doc_id] = {}
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS': []}
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                            else:
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                        else:
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS': []}
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                            else:
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)

                        continue

                    pred_word = ''
                    for occ_pred, ww, pp_aspect in occs_pred:
                        pred_sent_span = occ_pred[0]
                        if true_sent_span != pred_sent_span:
                            continue
                        ii += 1
                        # print(ii)
                        # print(true_sent_span, ',', pred_sent_span)
                        sent_found = True
                        true_aspect_span_start = occ_true[1]
                        true_aspect_span_end = occ_true[2]
                        pred_aspect_span_start = occ_pred[1]
                        pred_aspect_span_end = occ_pred[2]
                        pred_aspect_rule_num = occ_pred[4]

                        if (pred_aspect_span_start >= true_aspect_span_start and pred_aspect_span_end <= true_aspect_span_end) or \
                                (pred_aspect_span_start <= true_aspect_span_start and pred_aspect_span_end >= true_aspect_span_end):  # or \
                                # (pred_aspect_span_start <= true_aspect_span_start and pred_aspect_span_end >= true_aspect_span_start and pred_aspect_span_end <= true_aspect_span_end) or \
                            # (pred_aspect_span_start >= true_aspect_span_start and pred_aspect_span_start <= true_aspect_span_end and pred_aspect_span_end >= true_aspect_span_end):
                            is_tp = True
                            pred_word = ww
                            # print(ww)
                            is_fn = False
                            # tp += 1

                    if not sent_found:
                        continue
                    if is_fn:
                        fn += 1
                        # TODO - DUMPING DATA
                        if doc_id not in final_results.keys():
                            final_results[doc_id] = {}
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [],'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS' : []}
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                            else:
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                        else:
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [],'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS': []}
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)
                            else:
                                final_results[doc_id][sentence]['FN'].append(true_aspect.word)
                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY_FN'].append(cats)

                    if is_tp:
                        tp += 1
                        if true_aspect not in self.final_aspects_clusters:
                            self.final_aspects_clusters.append(true_aspect)
                        # TODO - DUMPING DATA
                        if doc_id not in final_results.keys():
                            final_results[doc_id] = {}

                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [],'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS': []}
                                final_results[doc_id][sentence]['TP'].append(true_aspect.word)
                                final_results[doc_id][sentence]['TP_EXT'].append(pred_word)
                                final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY'].append(cats)
                            else:
                                final_results[doc_id][sentence]['TP'].append(true_aspect.word)
                                final_results[doc_id][sentence]['TP_EXT'].append(pred_word)
                                final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY'].append(cats)


                        else:
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [],'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS': []}
                                final_results[doc_id][sentence]['TP'].append(true_aspect.word)
                                final_results[doc_id][sentence]['TP_EXT'].append(pred_word)
                                final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY'].append(cats)
                            else:
                                final_results[doc_id][sentence]['TP'].append(true_aspect.word)
                                final_results[doc_id][sentence]['TP_EXT'].append(pred_word)
                                final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                                if true_aspect.aspect_categories:
                                    cats = []
                                    for cat, count in true_aspect.aspect_categories.items():
                                        cats.append((cat, count))
                                    final_results[doc_id][sentence]['CATEGORY'].append(cats)

                                # TODO - DUMP RULE NUMBERS
                                #if true_aspect in aspect_rules.keys():
                                #    aspect_rules[true_aspect].append(pp_aspect)
                                #else:
                                #    aspect_rules[true_aspect] = [pp_aspect]
        # end

        sentences = []
        for pred_aspect in pred_aspects:
            pred_occ_list = pred_aspect.occ_list

            max_fp_doc = 1
            fp_count = 0
            for doc_id, occs_pred in pred_occ_list.items():
                occs_true = []
                for true_aspect in true_aspects:
                    true_occ_list = true_aspect.occ_list
                    for doc_id_true, occs_true_tmp in true_occ_list.items():
                        if doc_id == doc_id_true:
                            for occ in occs_true_tmp:
                                occs_true.append(occ)

                if len(occs_true) == 0:
                    continue

                true_sent_spans = []
                pred_sent_spans = []
                for op in occs_true:
                    span = op[0]
                    true_sent_spans.append(span)

                fp_sent_spans = []
                words = []

                is_sent_fp = False

                #is_fp = False
                for occ_pred in occs_pred:
                    is_fp = False
                    #is_tp = False
                    pred_sent_span = occ_pred[0]
                    sentence = occ_pred[3]
                    pred_aspect_rule_num = occ_pred[4]
                    if (pred_sent_span not in true_sent_spans) and (sentence not in sentences):
                        is_fp = True
                        fp_count += 1
                        #if fp_count > max_fp_doc:
                        #    continue

                        sentences.append(sentence)
                        #if sentence not in sentences:
                        #    is_sent_fp = True
                        #    sentences.append(sentence)
                        #else:
                        #    is_sent_fp = False

                        if pred_aspect.word not in words:
                            words.append(pred_aspect.word)
                        #fp += 1
                        #print(pred_aspect.word)
                        # TODO - DUMPING DATA

                    if is_fp:
                        # fp += 1
                        # if pred_aspect not in self.fp_aspects:
                        self.fp_aspects.append(pred_aspect)
                        if pred_aspect.word not in fp_words:
                            fp_words.append(pred_aspect.word)

                        if pred_aspect not in self.final_aspects_clusters :
                            self.final_aspects_clusters.append(pred_aspect)

                    # if is_fp:
                        if doc_id not in final_results.keys():
                            final_results[doc_id] = {}
                            print("SUCCESS - 1")
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS' : []}
                                for w in words:
                                    final_results[doc_id][sentence]['FP'].append(w)
                                    final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                            else:
                                for w in words:
                                    final_results[doc_id][sentence]['FP'].append(w)
                                    final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)

                        else:
                            if sentence not in final_results[doc_id].keys():
                                final_results[doc_id][sentence] = {'TP': [], 'TP_EXT': [], 'FP': [], 'FN': [],
                                                                   'CATEGORY': [], 'CATEGORY_FN': [], 'RULE_NUMS' : []}
                                for w in words:
                                    final_results[doc_id][sentence]['FP'].append(w)
                                    final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)


                            else:
                                for w in words:
                                    final_results[doc_id][sentence]['FP'].append(w)
                                    final_results[doc_id][sentence]['RULE_NUMS'].append(pred_aspect_rule_num)


        #end
        try:
            # BUG FIX - remove "chritmas present" which is tagged as a true aspect in annotation. 
            fn = fn - 1
            print("tp: ", tp)
            print("fp: ", len(self.fp_aspects))
            print("fn: ", fn)
            #print("fp_1", len(fp_words))
            # fp = 190
            fp = len(self.fp_aspects)
            # print("fp new: ", fp)
            precision = tp*100 / (tp + fp)
            recall = tp*100/ (tp + fn)
            f1 = 2*precision*recall / (precision + recall)

            print("Precision: %0.2f " % precision)
            print("Recall: %0.2f" % recall)
            print("F1-Score: %0.2f" % f1)
        except:
            pass

        #print(final_results)

        # WRITE TO CSV FILE
        #df = pd.DataFrame(final_results)
        rows = [['DOC_ID', 'Sentence', 'TP', 'TP_EXT', 'FN', 'FP', 'CATEGORY', 'CATEGORY_FN', 'RULE_NUMS']]
        for doc_id, results in final_results.items():
            for sent, result_list in results.items():
                row = [doc_id, sent, result_list['TP'], result_list['TP_EXT'], result_list['FN'], result_list['FP'],
                       result_list['CATEGORY'], result_list['CATEGORY_FN'], result_list['RULE_NUMS']]
                rows.append(row)

        df = pd.DataFrame(rows)
        pickle.dump(df, open("./output/extracted_aspects_2.pickle", 'wb'))
        df.to_csv("./output/extracted_aspects_2.csv")
        # pickle.dump(self.fp_aspects, open("fp.pickle", 'wb'))
        # RETURN THE TRUE EXTRACTED ASPECTS

        # todo - calculate the true positives again
        tp_count = 0
        tp_gold_count = 0
        fp_count = 0
        fn_count = 0
        for doc_id, results in final_results.items():
            for sent, result_list in results.items():
                tps = result_list['TP_EXT']
                tp_count += len(tps)
                tps_gold = result_list['TP']
                tp_gold_count += len(tps_gold)
                fp_count += len(result_list['FP'])
                fn_count += len(result_list['FN'])


        #print("TP extracted: ", tp_count)
        #print("TP gold count: ", tp_gold_count)
        #print("FP: ", fp_count)
        #print("FN: ", fn_count )

        #TODO - COMPARE RULES
        #rules_dic = {}
        #for t_aspect, p_aspects in aspect_rules.items():
        #    t_word = t_aspect.word
        #    ll = []
        #    for p_aspect in p_aspects:
        #        p_word = p_aspect.word
        #        rules = p_aspect.rule_nums
        #        ll.append((p_word, rules))

        #    rules_dic[t_word] = ll

        #df = pd.DataFrame.from_dict(rules_dic, orient="index")
        #df.to_csv("RULE_CLSUTERS.csv")

        return self.final_aspects_clusters


    def load_annotated_aspects(self):
        print("[INFO] Loading annotated dataset..")
        true_aspect_list = pickle.load(open("../data/annotated_aspects.pickle", 'rb'))
        return  true_aspect_list

    def load_IAC_lexicon(self):
        print("[INFO] Loading IAC lexicon..")
        self.iac_list = pickle.load(open("../data/IAC_lexicon.pickle", 'rb'))
        print("[INFO] IAC lexicon size: ", len(self.iac_list))

        return




def test_aspect_extraction_books():
    start_time = time.time()

    d = DataPreprocessor()
    reviews = d.load_book_reviews()

    ae1 = AspectExtractor(None, reviews)
    true_aspect_list = ae1.load_annotated_aspects()
    ae1.load_IAC_lexicon()
    pred_aspect_list = ae1.dependency_rule_based_features(true_aspect_list)

    end_time = time.time()

    print('[INFO] Aspect extraction finished in %0.2f sec' % (end_time - start_time))
    ae1.aspect_extraction_performance(pred_aspect_list, true_aspect_list, reviews)

    return



if __name__ == "__main__":
    test_aspect_extraction_books()
