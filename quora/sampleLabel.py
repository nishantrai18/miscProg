import math
import heapq
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import ADJ, ADJ_SAT, ADV, NOUN, VERB
from nltk.stem import WordNetLemmatizer
import time  # For computing running time

DEBUG = False


def upsert(a, key, inc):
    """
    Increments the value of a[key] by inc if key is present in a, else initializes a[key] with inc
    :param a: dictionary to work on
    :param key: key to update or create
    :param inc: increment or init value
    :return: None
    """
    if key in a:
        a[key] += inc
    else:
        a[key] = inc


class Question:
    def __init__(self, question_string=None):
        self.categories = []
        self.question = question_string
        self.tf = {}
        self.word_weights = {}
        self.non_stop_words = []


class Labeler:
    def __init__(self, datasize):
        self.all_words = {}  # known words, their counts and their index in context_matrix
        self.stopwords = frozenset(
            [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours',
             u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it',
             u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who',
             u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been',
             u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the',
             u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with',
             u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below',
             u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further',
             u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each',
             u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same',
             u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now'])
        self.dataset = []  # known questions
        self.dataset_size = datasize
        self.wnl = WordNetLemmatizer()
        self.word_context_matrix = []

    def save(self, category_string, question):
        """
        Creates a new Question and saves into the dataset
        :param category_string: A string of category IDs separated by a space
        :param question: question string to save
        :return: None
        """
        q = Question()
        q.categories = category_string.split(" ")
        q.question = question
        words = word_tokenize(question)[:-1]  # last char is a ?
        non_stop_words = []
        for w in words:
            if w.lower() not in self.stopwords:
                w = self.wnl.lemmatize(w, NOUN)
                non_stop_words.append(w)
        q.non_stop_words = non_stop_words

        for i, w in enumerate(non_stop_words):
            upsert(q.tf, w, 1)  # term frequency
            if w in self.all_words:
                self.all_words[w][0] += 1
            else:
                self.all_words[w] = [1, len(self.all_words)]
        self.dataset.append(q)

    def create_context_matrix(self):
        """
        Creates context matrix for finding similar words in our dataset
        context: 1 word to left and 1 word to the right of any word is its context
        Pre-Condition: self.all_words should be populated with all words.
        Pre-Condition: self.dataset should have all questions
        :return: word-word context matrix
        """
        total_words = len(self.all_words)
        for i in range(0, total_words):
            self.word_context_matrix.append([0] * total_words)

        for q in self.dataset:
            for i, w in enumerate(q.non_stop_words):
                self.update_context_for_word(i, q.non_stop_words)

    def update_context_for_word(self, i, words):
        """
        Updates the context of words[i]
        :param i: index of the word in words array, whose context has to be updated
        :param words: sentence as an array of words
        :return: update entries in context matrix
        """
        row_index = self.all_words[words[i]][1]
        if i > 0:
            lindex = self.all_words[words[i - 1]][1]
            self.word_context_matrix[row_index][lindex] += 1
        if i < len(words) - 1:
            rindex = self.all_words[words[i + 1]][1]
            self.word_context_matrix[row_index][rindex] += 1

    def prepare_question(self, question):
        """
        Create a question instance for a new question making it ready for querying with dataset
        :param question: question string to prepare
        :return: instance of Question class
        """
        q = Question(question)
        for w in word_tokenize(question)[:-1]:  # assuming last character of question is a ? mark
            w = self.wnl.lemmatize(w, NOUN)
            if w not in self.stopwords:
                upsert(q.tf, w, 1)  # case folding all words to lower
        self.compute_word_weights_for_q(q, self.dataset_size)
        return q

    def compute_word_weights(self):
        """
        Computes tf*idf vector for all questions in dataset
        :return: None
        """
        for q in self.dataset:
            self.compute_word_weights_for_q(q, self.dataset_size)

    def compute_word_weights_for_q(self, q, total_docs):
        """
        Computes tf_idf vector for the given question
        :param q: instance of Question class
        :param total_docs: total documents in dataset
        :return:
        """
        self.normalize_counts(q)  # so that longer questions will not have an added advantage
        for w in q.tf:
            df = 0  # default frequency for unknown words. Tip: Possible improvement
            if w in self.all_words:  # if we know the word
                df = self.all_words[w][0]
            idf = math.log(float(total_docs) / (1 + df), 10)
            q.word_weights[w] = q.tf[w] * idf

    def find_k_similar_questions(self, q, k, similarity):
        """
        Finds K most similar questions from the dataset to q
        :param q: question to find similar items to
        :param k: number of similar questions to q
        :param similarity: similarity function to use
        :return: list of K most similar questions from least to highest along with similarity scores
        [(score, qs_instance), (score, qs_instance)...]
        """
        min_heap = []
        for qs in self.dataset:
            s = similarity(qs, q)
            if len(min_heap) < k:
                heapq.heappush(min_heap, (s, qs))
            else:
                least_similar, item = heapq.heappop(min_heap)
                if s > least_similar:
                    heapq.heappush(min_heap, (s, qs))
                else:
                    heapq.heappush(min_heap, (least_similar, item))
        return [heapq.heappop(min_heap) for i in range(len(min_heap))]

    def find_k_categories(self, q, k, similarity):
        """
        Finds K categories for a question
        :param q: instance of Question to find categories
        :param k: Number of categories
        :param similarity: similarity function to use to compare with questions in dataset
        :return: list of categories from best match to least match
        """
        categories = []
        score_qs = self.find_k_similar_questions(q, k, similarity)
        for score, qs in reversed(score_qs):
            categories += qs.categories
            if len(categories) > k:
                break
        return categories[0:k]

    @staticmethod
    def cosine_similarity(q1, q2):
        """
        Cosine similarity between q1 and q2 question instances using their word_weight vectors
        :param q1: instance of Question class
        :param q2: instance of Question class
        :return: similarity between q1 and q2
        """
        numerator = 0
        q1_sum_squares = 0
        q2_sum_squares = 0
        for w in q1.word_weights:
            q1_sum_squares += (q1.word_weights[w] * q1.word_weights[w])
            if w in q2.word_weights:
                numerator += (q1.word_weights[w] * q2.word_weights[w])
        for w in q2.word_weights:
            q2_sum_squares += (q2.word_weights[w] * q2.word_weights[w])
        denominator = math.sqrt(q1_sum_squares) * math.sqrt(q2_sum_squares)
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def normalize_counts(q):
        """
        Normalizes the term frequency counts of the question, q
        :param q: instance of Question
        :return: None
        """
        total = 0
        for c in q.tf.values():
            total += (c * c)
        normalization_factor = math.sqrt(total)
        for w in q.tf:
            q.tf[w] /= normalization_factor


def test():
    l = Labeler(3)
    l.save("3 1 2 4", "What is the meaning of life?")
    l.save("7 1 2 5 8 9 11 15", "What is Quora?")
    l.save("2 14 178", "What are the best Google calendar hacks?")
    l.create_context_matrix()
    l.compute_word_weights()
    print l.find_k_categories(l.prepare_question("What is Google Calendar?"), 3, Labeler.cosine_similarity)
    print l.find_k_categories(l.prepare_question("Is Quora has meaning for life?"), 3, Labeler.cosine_similarity)


def main():
    t, e = [int(i) for i in raw_input().split(" ")]  # # of categorised questions, # of un-categorised questions
    l = Labeler(t)
    for i in range(0, t):
        category_string_with_count = raw_input()  # count category_id category_id category_id...
        category_string = category_string_with_count[category_string_with_count.index(" ") + 1:]
        question_string = raw_input()
        l.save(category_string, question_string)
    l.create_context_matrix()
    #print(l.word_context_matrix)
    l.compute_word_weights()
    for i in range(0, e):
        print " ".join(l.find_k_categories(l.prepare_question(raw_input()), 10, Labeler.cosine_similarity))


if __name__ == '__main__':
    start_time = time.time()
    main()
    #test()
    if DEBUG:
        print("--- %s seconds ---" % (time.time() - start_time))

# TODO
# 7) POS for Lemmatization
# 2) Word-Context Matrix
# 6) What to do with 0 match questions?
# 3) Consider Synonyms
# 5) Try Bayesian Estimation also (PPT 16 - Slides #31 - #33)



# 1) Lemmatization

# 4) After finding K similar questions, choose topics which are most frequent in all these K