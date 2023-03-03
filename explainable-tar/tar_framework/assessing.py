# coding=utf-8
import sys
import os
import json
import copy
import numpy as np
from collections import defaultdict
from tar_framework.run_utilities import LOGGER, REL


class DataLoader(object):
    def __init__(self, query_file, qrel_file, doc_id_file, doc_text_file):
        """
        Load data.
        @param query_file: e.g. {"id": 1, "query": , "title": }
        @param qrel_file: TREC qrel format, each line is an example, e.g. qid 0 did 1
        @param doc_id_file: stores the list of document ids from which to run the TAR process, each line is an id
        @param doc_text_file: stores the texts of document ids in doc_id_file, each line is a json format string, e.g. {"id": 1, "title": , "content": }
        """
        self.title = self.read_title(query_file)
        self.document_id_to_label = self.read_qrels(qrel_file)
        self.document_ids = self.read_doc_ids(doc_id_file)
        self.document_id_to_text = self.read_doc_texts(doc_text_file)

        self.pseudo_document_id = 'pseudo_did'
        self.pseudo_text = self.title
        self.pseudo_label = REL

        LOGGER.info('{} DataLoader.__init__ is done.'.format(os.path.basename(query_file)))

    @staticmethod
    def read_title(query_file):
        with open(query_file, 'r', encoding='utf8') as f:
            entry = json.loads(f.read())
            return entry['title']

    @staticmethod
    def read_qrels(qrel_file):
        dct = {}
        with open(qrel_file, 'r', encoding='utf8') as f:
            for line in f:
                if len(line.split()) != 4:
                    continue
                topic_id, dummy, doc_id, rel = line.split()
                dct[doc_id] = int(rel)
        return dct

    @staticmethod
    def read_doc_ids(doc_id_file):
        dids =[]
        with open(doc_id_file, 'r', encoding='utf8') as f:
            for line in f:
                dids.append(line.strip())
        return dids

    @staticmethod
    def read_doc_texts(doc_text_file):
        dct = {}
        with open(doc_text_file, 'r', encoding='utf8') as f:
            for line in f:
                # entry = {'id': doc_id, 'title': subject, 'content': content}
                if len(line) == 1:
                    continue
                entry = json.loads(line)
                doc_id = entry['id']
                text = entry['title'] + ' ' + entry['content']
                dct[doc_id] = text
        return dct

    @staticmethod
    def read_doc_texts_2_list(doc_text_file):
        """
        Only used in autotar_method.
        @param doc_text_file:
        @return:
        """
        with open(doc_text_file, 'r', encoding='utf8') as f:
            for line in f:
                entry = json.loads(line)
                text = entry['title'] + ' ' + entry['content']
                yield text

    def get_title(self):
        return self.title

    def get_document_id_to_label(self):
        return self.document_id_to_label

    def get_complete_document_ids_with_pseudo_document(self):
        return self.document_ids + [self.pseudo_document_id]

    def get_complete_document_texts_with_pseudo_document(self):
        return [self.document_id_to_text[did] for did in self.document_ids] + [self.pseudo_text]

    def get_complete_document_ids(self):
        return self.document_ids

    def get_complete_texts(self):
        return [self.document_id_to_text[did] for did in self.document_ids]

    def get_complete_labels(self):
        return [self.document_id_to_label[did] for did in self.document_ids]

    def get_rel_label(self, did):
        return self.document_id_to_label[did]

    def get_document_count(self):
        return len(self.document_ids)

    def get_relevant_document_count(self):
        return len(list(filter(lambda did: self.document_id_to_label[did] == REL, self.document_ids)))


class Assessor(DataLoader):
    """
    Manager the assessment module of the TAR framework.
    """
    def __init__(self,query_file, qrel_file, doc_id_file, doc_text_file):
        super().__init__(query_file, qrel_file, doc_id_file, doc_text_file)

        self.assessed_document_ids = []
        self.unassessed_document_ids = copy.copy(self.document_id_to_label)
        self.assessment_state = defaultdict(lambda: False)

    def get_training_data(self, number_of_training_docs):
        """
        Provide training data for training ranker
        :param type:
        :return:
        """
        assessed_document_ids = self.get_assessed_document_ids()

        population = self.get_unassessed_document_ids()
        number_of_training_docs = min(len(population), number_of_training_docs)
        temp_document_ids = list(np.random.choice(a=population, size=number_of_training_docs, replace=False))  # unique random elements

        document_ids = [self.pseudo_document_id] + assessed_document_ids + temp_document_ids
        labels = [self.pseudo_label] + [self.document_id_to_label[did] for did in assessed_document_ids] + len(temp_document_ids)*[0]

        assert len(document_ids) == len(labels)
        return document_ids, labels

    def update_assessment(self, document_ids):

        for document_id in document_ids:
            if self.assessment_state[document_id] is False:
                self.assessed_document_ids.append(document_id)
                self.unassessed_document_ids.pop(document_id)
                self.assessment_state[document_id] = True
        return

    def assess_state_check_func(self):
        def func(document_id):
            return self.assessment_state[document_id]
        return func

    def get_top_assessed_document_ids(self, ranked_document_ids, threshold):
        document_count = 0
        top_document_ids = []
        for document_id in ranked_document_ids:
            if self.assessment_state[document_id] is False:
                top_document_ids.append(document_id)
                document_count += 1
            if document_count >= threshold:
                break
        return top_document_ids

    def get_assessed_document_ids(self):
        return self.assessed_document_ids

    def get_assessed_count(self):
        return len(self.assessed_document_ids)

    def get_assessed_relevant_document_ids(self):
        assessed_document_ids = self.get_assessed_document_ids()
        return list(filter(lambda document_id: self.document_id_to_label[document_id] == REL, assessed_document_ids))

    def get_assessed_relevant_count(self):
        assessed_document_ids = self.get_assessed_document_ids()
        return len(list(filter(lambda document_id: self.document_id_to_label[document_id] == REL, assessed_document_ids)))

    def get_unassessed_document_ids(self):
        return list(self.unassessed_document_ids.keys())

    def get_unassessed_document_count(self):
        return len(self.unassessed_document_ids.keys())

    def get_assessed_state(self):
        return self.assessment_state


if __name__ == '__main__':
    topic_id = 'CD007394'
    mdir = os.path.join(os.getcwd(), 'data/clef2017')
    query_file = os.path.join(mdir, 'topics', topic_id)
    qrel_file = os.path.join(mdir, 'qrels', topic_id)
    doc_id_file = os.path.join(mdir, 'docids', topic_id)
    doc_text_file = os.path.join(mdir, 'doctexts', topic_id)

    data = DataLoader(query_file, qrel_file, doc_id_file, doc_text_file)
    pass