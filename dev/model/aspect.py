

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

        #print(aspect_category)
        if doc_id not in self.occ_list.keys():
            self.occ_list[doc_id] = []
            self.occ_list[doc_id].append([line, start, end, sent,rule_num])
        else:
            self.occ_list[doc_id].append([line, start, end, sent, rule_num])
            # print(self.word,"  " ,self.occ_list)

        self.doc_freq = len(self.occ_list)
        self.is_explicit = is_explict

        if not aspect_category:
            return

        if not self.aspect_categories:
            self.aspect_categories = {}

        if aspect_category not in self.aspect_categories.keys():
            self.aspect_categories[aspect_category] = 1
            #print(self.aspect_categories)
        else:
            self.aspect_categories[aspect_category] += 1

        self.rule_nums.append(rule_num)

        return






