import pandas as pd
import spacy
import collections
import numpy as np
import time
import pickle
import nltk
import operator
import math

import sys
sys.path.append('../')

import model.aspect_extractor as ae

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import  pairwise_distances
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en")
nlp_sim = spacy.load('en_core_web_lg')


class Feature:
    def __init__(self, word, is_explicit, true_category):
        self.common_words = []
        self.word = word
        self.is_explict = is_explicit
        self.true_category = true_category
        self.pred_category = None
        self.frequency = 0
        self.feature_id = 0
        self.matrix_id = 0
        self.sim_g_vector = None
        self.sim_t_vector = None
        self.sim_gt_vector = None

    def set_frequency(self, frequncy):
        self.frequency = frequncy

    def set_pred_category(self, pred_category):
        self.pred_category = pred_category

    def set_feature_id(self, id):
        self.feature_id = id

    def set_sim_g_vector(self, vector):
        self.sim_g_vector = vector

    def set_sim_t_vector(self, vector):
        self.sim_t_vector = vector

    def set_sim_gt_vector(self, vector):
        self.sim_gt_vector = vector

    def init_variables(self):
        self.common_words = []
        return

    def add_sharing_words(self, word_list):
        # print(word_list)
        for w in word_list:
            # print(w)
            if w not in self.common_words:
                self.common_words.append(w)
        # print(self.common_words)


class AspectCluster:

    def __init__(self):
        self.aspect_list = []
        self.seed_aspects = []
        self.nonseed_aspcets = []
        self.aspect_clusters = []
        self.inverted_index = []
        self.start_time = time.time()

        # hyper parameter3
        self.SIM_THRESHOLD = 0.030
        self.W_G = 1

    # LOADING DATA


    def load_book_aspect_data(self):

        csv_file = "./data/dep_clusters.csv"
        df = pd.read_csv(csv_file)
        #df = pd.read_excel(xls_file)
        df_data = df.to_dict()

        word_list = []
        feature_list = []
        for category, words in df_data.items():
            for k, w_ in words.items():
                if type(w_) is float or type(w_) is int:
                    continue

                w_list = w_.split('---')

                for w in w_list:
                    w = w.strip().rstrip()
                    word_list.append(w)
                    print(w)
                    is_explicit = False
                    nlp_w = nlp(w)
                # print(w)

                    for token in nlp_w:
                        if token.pos_ == "NOUN":
                            is_explicit = True
                            break

                    feature = Feature(w, is_explicit, category)
                    feature_list.append(feature)

        freq_words = collections.Counter(word_list)
        for word, freq in freq_words.items():

            for feature in feature_list:
                if feature.word == word:
                    feature.set_frequency(freq)

        self.aspect_list = feature_list
        for index, feature in enumerate(self.aspect_list):
            feature.set_feature_id(index)

        count = 0
        for aspect_1 in self.aspect_list:
            count += 1
            print(count, '/', len(self.aspect_list))
            aspect1_words = []
            words_pos = pos_tag(nltk.word_tokenize(aspect_1.word))
            for wp in words_pos:
                pos = self.find_pos(wp[1])
                lemma = lemmatizer.lemmatize(wp[0], pos)
                aspect1_words.append(lemma)
            for aspect_2 in self.aspect_list:
                aspect2_words = []
                words_pos = pos_tag(nltk.word_tokenize(aspect_2.word))
                for wp in words_pos:
                    pos = self.find_pos(wp[1])
                    lemma = lemmatizer.lemmatize(wp[0], pos)
                    aspect2_words.append(lemma)
                common_words = list(set(aspect1_words).intersection(aspect2_words))
                if common_words:
                    aspect_1.add_sharing_words(common_words)

        return feature_list


    # UTILITY
    def pickle_features(self):
        # pickle.dump(self.aspect_list, open("./pickle/cellphone_features.pickle", 'wb'))
        pickle.dump(self.aspect_list, open("./data/book_features_dep.pickle", 'wb'))

        return

    def pickle_sim_g_matrix(self):
        # pickle.dump(self.sim_g_matrix, open("./pickle/sim_g_cell_matrix.pickle", 'wb'))
        pickle.dump(self.sim_g_matrix, open("./data/sim_g_book_matrix_dep.pickle", 'wb'))

        return

    def find_pos(self, pos):
        nouns = ['NN', 'NNS', 'NNP', 'NNPS']
        verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        adverbs = ['RB', 'RBR', 'RBS']
        adjectives = ['JJ', 'JJR', 'JJS']

        if pos in nouns:
            return 'n'
        if pos in verbs:
            return 'v'
        if pos in adjectives:
            return 'a'
        return 'n'


    def load_feature_pickle(self):
        print("Loading feature pickle..")
        aspect_list = pickle.load(open("../data/book_features_dep.pickle", 'rb'))
        # aspect_list = pickle.load(open("./pickle/cellphone_features.pickle", 'rb'))
        for asect in aspect_list:
            ass = Feature(asect.word, asect.is_explict, asect.true_category)
            ass.set_feature_id(asect.feature_id)
            # print(ass.feature_id)
            ass.set_frequency(asect.frequency)
            ass.matrix_id = asect.matrix_id
            # print(asect.frequency)
            ass.set_sim_g_vector(asect.sim_g_vector)
            self.aspect_list.append(ass)

        for aspect_1 in self.aspect_list:
            aspect1_words = []
            words_pos = pos_tag(nltk.word_tokenize(aspect_1.word))
            for wp in words_pos:
                pos = self.find_pos(wp[1])
                lemma = lemmatizer.lemmatize(wp[0], pos)
                aspect1_words.append(lemma)
            for aspect_2 in self.aspect_list:
                aspect2_words = []
                words_pos = pos_tag(nltk.word_tokenize(aspect_2.word))
                for wp in words_pos:
                    pos = self.find_pos(wp[1])
                    lemma = lemmatizer.lemmatize(wp[0], pos)
                    aspect2_words.append(lemma)
                common_words = list(set(aspect1_words).intersection(aspect2_words))
                if common_words:
                    # aspect_1.common_words.append(common_words)
                    aspect_1.add_sharing_words(common_words)

        return self.aspect_list

    def load_sim_g_pickle(self):
        # self.sim_g_matrix = pickle.load(open("./pickle/sim_g_cell_matrix.pickle", 'rb'))
        print("Loading sim_g pickle ..")
        self.sim_g_matrix = pickle.load(open("../data/sim_g_book_matrix_dep.pickle", 'rb'))
        print(self.sim_g_matrix.min(), " : ", self.sim_g_matrix.max())
        #self.sim_g_matrix = (self.sim_g_matrix - self.sim_g_matrix.min()) / (self.sim_g_matrix.max() - self.sim_g_matrix.min())
        return

    def get_feature_from_id(self, feature_id):
        return self.aspect_list[feature_id]

    # SIMILARITY FUNCTIONS
    def spacy_similairty(self, i, j):

        #feature_i = self.get_feature_from_id(int(i[0]))
        #feature_j = self.get_feature_from_id(int(j[0]))
        # doc_i = nlp_sim(feature_i.word)
        # doc_j = nlp_sim(feature_j.word)
        feature_i = self.feature_word_array[int(i[0])]
        feature_j = self.feature_word_array[int(j[0])]
        doc_i = nlp_sim(feature_i)
        doc_j = nlp_sim(feature_j)
        similarity = doc_i.similarity(doc_j)

        #if similarity < 0:
        if similarity > 0.8 and similarity < 1.0:
            print(int(i[0]), ':',  feature_i, ", ", int(j[0]), ":", feature_j,  " - ", similarity)
        return similarity

    def generate_feature_sim_g_vecotrs(self):

        start_time = time.time()
        self.feature_word_array = []
        for aspect in self.aspect_list:
            word = aspect.word
            if word not in self.feature_word_array:
                self.feature_word_array.append(word)

        for aspect in self.aspect_list:
            aspect_word = aspect.word
            for index, word in enumerate(self.feature_word_array):
                if aspect_word == word:
                    aspect.matrix_id = index

        print("Array size : ", len(self.feature_word_array))

        # darry = np.array([[x.feature_id] for x in self.aspect_list])
        darry = np.array([[index] for index, w in enumerate(self.feature_word_array)])
        sim_g_vectors = pairwise_distances(darry, metric=self.spacy_similairty, n_jobs=-1)

        print(sim_g_vectors.min()," , " , sim_g_vectors.max())
        sim_g_vectors = (sim_g_vectors - sim_g_vectors.min())/(sim_g_vectors.max() - sim_g_vectors.min())
        print(sim_g_vectors.min(),' , ' ,sim_g_vectors.max())

        for aspect in self.aspect_list:
            matrix_id = aspect.matrix_id
            sim_g_vector = sim_g_vectors[matrix_id]
            aspect.set_sim_g_vector(sim_g_vector)

        for index, vector in enumerate(sim_g_vectors):
            feature = self.aspect_list[index]
            feature.set_sim_g_vector(vector)

        print("sim_g_vectors for features generated in %0.2f sec" % (time.time() - start_time))
        self.pickle_features()
        return

    def generate_similarity_g_matrix(self):
        darry = np.array([x.sim_g_vector for x in self.aspect_list])
        self.sim_g_matrix = pairwise_distances(darry, metric='cosine', n_jobs=-1)
        print(self.sim_g_matrix.min(), self.sim_g_matrix.max())
        self.sim_g_matrix = 1 - self.sim_g_matrix
        self.sim_g_matrix = (self.sim_g_matrix -self.sim_g_matrix.min() ) / (self.sim_g_matrix.max() - self.sim_g_matrix.min())
        self.pickle_sim_g_matrix()

        print(self.sim_g_matrix.min(), self.sim_g_matrix.max())
        return



    # CLUSTERING DISTANCE FUNCTIONS
    def weighted_similarity(self, feature_i, feature_j):
        #i = feature_i.feature_id
        #j = feature_j.feature_id

        i = feature_i.matrix_id
        j = feature_j.matrix_id

        #sim = 1 - feature_i.sim_g_vector[j]
        # print( feature_i.word, " --", feature_i.sim_g_vector, " , ",  feature_j.word, " --", feature_j.sim_g_vector)
        sim_g = self.sim_g_matrix[i][j]
        sim = 1 - self.W_G*sim_g
        #print(feature_i.word, ' , ', feature_j.word, " , ", sim , " ,  ", sim_g)

        # print(i, " : ", j, " : ", sim)
        # Check for similar  sharing words
        common_words_i = feature_i.common_words
        common_words_j = feature_j.common_words

        intersec = len(list(set(common_words_i).intersection(common_words_j)))
        union = len(list(set(common_words_i).union(common_words_j)))

        y = (union - intersec)
        if y == 0:
            sim = 0
        else:
            x = -(intersec/(union - intersec))
            sim = sim*math.exp(x)

        return sim

    def average_cluster_distance(self, cluster_i, cluster_j):
        dist = 0
        is_explict = False
        for feature_i in cluster_i.aspects:
            if feature_i.is_explict:
                is_explict = True
            for feature_j in cluster_j.aspects:
                if feature_j.is_explict:
                    is_explict = True
                sim = self.weighted_similarity(feature_i, feature_j)
                dist += sim

        dist = dist/(len(cluster_i.aspects)*len(cluster_j.aspects))
        # print(len(cluster_i.aspects)*len(cluster_j.aspects))
        return dist, is_explict

    def rep_cluster_distance(self, cluster_i, cluster_j):
        sorted_i = sorted(cluster_i.aspects, key=lambda aspect: aspect.frequency, reverse=True)
        sorted_j = sorted(cluster_j.aspects, key=lambda aspect: aspect.frequency, reverse=True)
        sim = self.weighted_similarity(sorted_i[0], sorted_j[0])
        return sim

    def seed_distance_function(self, i, j):
        i = int(i[0])
        j = int(j[0])

        if i == j:
            return 1

        cluster_i = self.aspect_clusters[i]
        cluster_j = self.aspect_clusters[j]


        distance_avg, is_explict = self.average_cluster_distance(cluster_i, cluster_j)
        distance_rep = self.rep_cluster_distance(cluster_i, cluster_j)

        if not is_explict: #Constraint 2
            distance = 1
        else:
            distance = max(distance_avg, distance_rep)
        return distance

    def nonseed_distance_function(self, i, j):
        i = int(i[0])
        j = int(j[0])

        cluster_i = self.aspect_clusters[i]
        cluster_j = self.nonseed_clusters[j]

        distance_avg, is_explict = self.average_cluster_distance(cluster_i, cluster_j)
        distance_rep = self.rep_cluster_distance(cluster_i, cluster_j)

        #if not is_explict:
        #    distance = 1
        #else:
        distance = min(distance_avg, distance_rep)

        return distance

    # CLUSTERING
    def generate_seed_features(self):

        sorted_features = sorted(self.aspect_list, key=lambda x: x.frequency, reverse=True)
        seed_limit = 1.0*len(self.aspect_list)
        for index, feature in enumerate(sorted_features):
            if index < seed_limit:
                print(feature.feature_id)
                self.seed_aspects.append(feature)
            else:
                self.nonseed_aspcets.append(feature)
        return

    def find_seed_clusters(self):
        sorted_clusters = sorted(self.aspect_clusters, key=lambda x: len(x.aspects), reverse=True)
        self.nonseed_clusters = []
        seed_clusters = []
        seed_limit = 0.2*len(sorted_clusters)
        for index, cluster in enumerate(sorted_clusters):
            if index < seed_limit:
                seed_clusters.append(cluster)
            else:
                self.nonseed_clusters.append(cluster)

        self.aspect_clusters = seed_clusters
        return

    def merge_two_clusters(self, cluster_i, cluster_j):

        for feature in cluster_j.aspects:
            cluster_i.add_aspect(feature)

        return

    def merge_clusters(self):

        print("Merge clusters ..")
        self.generate_seed_features()
        print("Number of features: ", len(self.seed_aspects))
        for index, feature in enumerate(self.seed_aspects):
            cluster = Cluster(index)
            cluster.add_aspect(feature)
            self.aspect_clusters.append(cluster)

        continue_clustering = True
        clust_round = 0
        check_count = 0
        while continue_clustering:
            if check_count > 10:
                continue_clustering = False
            darray = np.array([[index] for index, cluster in enumerate(self.aspect_clusters)])
            if not darray.tolist():
                continue

            distance_matrix = pairwise_distances(darray, metric=self.seed_distance_function, n_jobs=-1)

            min_distance = distance_matrix.min()
            if min_distance <= self.SIM_THRESHOLD: # Constraint 1
                continue_clustering = True
                min_index = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
                i = min_index[0]
                j = min_index[1]
                cluster_i = self.aspect_clusters[i]
                cluster_j = self.aspect_clusters[j]
                self.merge_two_clusters(cluster_i, cluster_j)
                self.aspect_clusters.remove(cluster_j)
            else:
                check_count += 1

            # continue_clustering = False
            clust_round += 1
            print("ROUND %d, time: %0.2f min, distance: %0.2f" % (clust_round, (time.time() - self.start_time)/60, min_distance))

        print("[INFO] Seed clustering finished in %0.2f min" % ((time.time() - self.start_time)/60))
        return

    def merge_clusters_2(self):

        print("Number features: ", len(self.nonseed_aspcets))
        seed_index = len(self.seed_aspects)
        self.nonseed_clusters = []
        for index, feature in enumerate(self.nonseed_aspcets):
            cluster = Cluster(index+seed_index)
            cluster.add_aspect(feature)
            self.nonseed_clusters.append(cluster)

        print("Merge cluster ..")
        #self.find_seed_clusters()
        continue_clustering = True
        clust_round = 0
        while continue_clustering:
            continue_clustering = False
            darray_1 = np.array([[index] for index, cluster in enumerate(self.aspect_clusters)])
            darray_2 = np.array([[index] for index, cluster in enumerate(self.nonseed_clusters)])

            # print(darray_1.tolist())
            # print(darray_2.tolist())
            # print("--------------")
            # STOP IF ONE OF THE ARRAY IS EMPTY
            if not darray_1.tolist() or not darray_2.tolist():
                continue

            distance_matrix = pairwise_distances(darray_1, darray_2, metric=self.nonseed_distance_function, n_jobs=-1)
            min_distance = distance_matrix.min()
            if min_distance < self.SIM_THRESHOLD:
                continue_clustering = True
                min_index = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
                i = min_index[0]
                j = min_index[1]
                cluster_i = self.aspect_clusters[i]
                cluster_j = self.nonseed_clusters[j]
                self.merge_two_clusters(cluster_i, cluster_j)
                self.nonseed_clusters.remove(cluster_j)

            clust_round += 1
            print("ROUND %d, time: %0.2f min, distance: %0.2f" % (clust_round, (time.time() - self.start_time)/60, min_distance))

        print("[INFO] None clustering finished in %0.2f min" % ((time.time() - self.start_time)/60))

        for cluster in self.nonseed_clusters:
            self.aspect_clusters.append(cluster)

        return

    def update_predicted_cluster_id(self):
        for index, cluster in enumerate(self.aspect_clusters):
            for feature in cluster.aspects:
                feature.set_pred_category(index)
        return

    def run_similarity_g(self):
        self.load_book_aspect_data()
        self.generate_feature_sim_g_vecotrs()
        self.generate_similarity_g_matrix()
        return


    def run_clustering(self):
        distance_thrshold = self.SIM_THRESHOLD
        print("[INFO] Clustering distance threshold : ", distance_thrshold)
        self.load_feature_pickle()
        self.load_sim_g_pickle()
        self.merge_clusters()
        self.merge_clusters_2()
        self.clustering_performance()

        return

    # CLUSTERING PERFORMANCE
    def clustering_performance(self):

        self.update_predicted_cluster_id()
        true_clusters = []
        pred_clusters = []


        for feature in self.aspect_list:
            freq = feature.frequency
            print(freq)
            true_clusters.append(feature.true_category)
            pred_clusters.append(feature.pred_category)


        aspect_categories = {}
        for feat in self.aspect_list:
            category = feat.pred_category
            word = feat.word
            if category not in aspect_categories.keys():
                aspect_categories[category] = [word]
            else:
                aspect_categories[category].append(word)


        df = pd.DataFrame.from_dict(aspect_categories, orient='index')
        df = df.T
        df.to_csv("./output/predicted_clusters_0.03_dep.csv")

        for fe in self.aspect_list:
            print(fe.true_category, ' : ',fe.pred_category, ' - ',fe.word)

        rand_index = adjusted_rand_score(true_clusters, pred_clusters)
        homo = metrics.homogeneity_score(true_clusters, pred_clusters)
        com = metrics.completeness_score(true_clusters, pred_clusters)


        print("True clusters: %d, Pred clusters: %d, Rand Index: %0.3f, Homo: %0.3f, Compl: %0.3f " % (len(set(true_clusters)), len(set(pred_clusters)), rand_index*100, homo*100, com*100))

        pickle.dump(self.aspect_list, open('./output/aspect_list_0.03_dep.pickle', 'wb'))

        return


class Cluster:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.aspects = []

    def add_aspect(self, aspect):
        self.aspects.append(aspect)

if __name__ == '__main__':

    ac = AspectCluster()
    ac.run_clustering()





