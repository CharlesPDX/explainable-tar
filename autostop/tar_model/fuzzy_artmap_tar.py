# coding=utf-8

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
print(sys.path)

import csv
# import math
import numpy as np
from operator import itemgetter
# from scipy.stats import norm
# from sklearn.preprocessing import MinMaxScaler
from tar_framework.assessing import Assessor
from tar_framework.ranking import Ranker
from tar_model.utils import *
from tar_framework.utils import *

def fuzzy_artmap_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                stopping_percentage=1.0, stopping_recall=None,  # autostop parameters
                min_df=2,
                random_state=0):
    """
    TAR implementation using Fuzzy ARTMAP as the classifier.
    See
    @param data_name:
    @param topic_set:
    @param topic_id:
    @param stopping_percentage:
    @param stopping_recall:
    @param random_state: random seed
    @return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'fam' + '-'    
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'

    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker(model_type="fam", random_state=random_state, min_df=min_df)
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts)
    ranker.set_features_by_name('complete_dids', complete_dids)

    # local parameters
    stopping = False
    t = 0
    batch_size = 100
    temp_doc_num = 100
    
    # starting the TAR process
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(("iteration", "batch_size", "total_num", "sampled_num", "total_true_r", "running_true_r", "ap", "running_true_recall"))
        while not stopping:
            t += 1
            LOGGER.info('TAR: iteration={}'.format(t))
            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train(train_features, train_labels)

            test_features = ranker.get_feature_by_did(assessor.get_unassessed_dids())
            scores = ranker.predict(test_features)

            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, _ = zip(*zipped)

            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_dids(ranked_dids, batch_size)
            assessor.update_assess(selected_dids)

            # statistics
            sampled_num = assessor.get_assessed_num()
            # sampled_percentage = sampled_num/total_num
            running_true_r = assessor.get_assessed_rel_num()
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            # batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow((t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall))

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    shown_dids = assessor.get_assessed_dids()
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                     exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    LOGGER.info('TAR is finished.')

    return

if __name__ == '__main__':
    # test_detect_knee()

    # data_name = 'clef2017'
    # topic_id = 'CD008081'
    # topic_set = 'test'
    data_name = '20newsgroups'
    topic_id = 'alt.atheism'
    topic_set = 'alt.atheism'
    query_file = os.path.join(PARENT_DIR, 'data', data_name, 'topics', topic_id)
    qrel_file = os.path.join(PARENT_DIR, 'data', data_name, 'qrels', topic_id)
    doc_id_file = os.path.join(PARENT_DIR, 'data', data_name, 'docids', topic_id)
    doc_text_file = os.path.join(PARENT_DIR, 'data', data_name, 'doctexts', topic_id)

    fuzzy_artmap_method(data_name, topic_id, topic_set,query_file, qrel_file, doc_id_file, doc_text_file)