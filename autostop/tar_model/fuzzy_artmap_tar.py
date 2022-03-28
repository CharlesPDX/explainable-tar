# coding=utf-8
from datetime import datetime
# from logging import Logger
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/measures'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/seeker'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/tar_framework'))
print(sys.path)

import gc

import json
import csv
# import math
import random
import numpy as np
from operator import itemgetter

import keepsake
import tornado.ioloop

# from scipy.stats import norm
# from sklearn.preprocessing import MinMaxScaler
from tar_framework.assessing import Assessor
from tar_framework.ranking import Ranker, VectorizerType
from tar_model.utils import *
from tar_framework.utils import *

from trec_eval.tar_eval import main as eval

async def fuzzy_artmap_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                stopping_percentage=1.0, stopping_recall=None,  # autostop parameters
                random_state=0,
                vectorizer_params=None,
                vectorizer_type=VectorizerType.tf_idf,
                classifier_params=None,
                **kwargs):
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
    # np.random.seed(random_state)

    # model named with its configuration
    model_name = 'fam' + '-'    
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'

    LOGGER.info('Model configuration: {}.'.format(model_name))
    LOGGER.debug('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker(**classifier_params)
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts, vectorizer_type=vectorizer_type, corpus_name=data_name, vectorizer_params=vectorizer_params)
    ranker.set_features_by_name('complete_dids', complete_dids)
    LOGGER.info("Caching corpus")
    await ranker.cache_corpus_in_model(complete_dids)
    LOGGER.info("Caching complete")

    # local parameters
    stopping = False
    t = 0
    batch_size = 100
        
    # starting the TAR process
    start_time = datetime.now()
    # perform initial model training, with some positive examples
    shuffled_doc_ids = random.sample(assessor.get_complete_dids(), len(assessor.get_complete_dids()))
    initial_positive_doc_ids = list(filter(lambda doc_id: assessor.did2label[doc_id] == REL, shuffled_doc_ids))[:10]
    initial_negative_doc_ids = list(filter(lambda doc_id: assessor.did2label[doc_id] != REL, shuffled_doc_ids))[:90]
    initial_training_doc_ids = list(initial_positive_doc_ids)
    initial_training_doc_ids.extend(initial_negative_doc_ids)
    initial_training_labels = list(len(initial_positive_doc_ids) * [1])
    initial_training_labels.extend(len(initial_negative_doc_ids) * [0])
    initial_training_features = ranker.get_feature_by_did(initial_training_doc_ids)
    
    LOGGER.info(f"starting initial training")
    await ranker.train(initial_training_features, initial_training_labels)
    LOGGER.info(f"initial training complete - {len(initial_training_doc_ids):,} documents")

    last_r = 0
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(("iteration", "batch_size", "total_num", "sampled_num", "total_true_r", "running_true_r", "ap", "running_true_recall", "sampled_percentage"))
        gc.disable()
        while not stopping:
            iteration_start_time = datetime.now()
            t += 1
            gc_stats = gc.get_stats()
            LOGGER.info(f'TAR: iteration={t}, {gc_stats[0]["collections"]}, {gc_stats[1]["collections"]}, {gc_stats[2]["collections"]}')

            unassessed_document_ids = assessor.get_unassessed_dids()
            # test_features = ranker.get_feature_by_did(unassessed_document_ids)
            # scores = ranker.predict_with_doc_id(test_features, unassessed_document_ids)
            scores = await ranker.predict_with_doc_id(unassessed_document_ids)
            
            zipped = sorted(scores, key=itemgetter(0), reverse=True)
            if len(zipped) > 0:
                _, ranked_dids = zip(*zipped)
            else:
                ranked_dids = []

            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_dids(ranked_dids, batch_size)
            assessor.update_assess(selected_dids)

            # statistics
            sampled_num = assessor.get_assessed_num()
            sampled_percentage = sampled_num/total_num
            running_true_r = assessor.get_assessed_rel_num()
            running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

            # update parameters
            # batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            csvwriter.writerow((t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall, sampled_percentage))
            f.flush()

            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True
            if running_true_r == total_true_r:
                LOGGER.info(f"Stopping - all relevant documents found")
                stopping = True
            if len(zipped) == 0 or zipped[0][0] == 0 or zipped[0][0] == "0":
                if running_true_r == last_r:
                    LOGGER.info("No more relevant documents found")
                    stopping = True

            last_r = running_true_r

            # train model with new assessments
            if not stopping:
                LOGGER.info("Starting assessed document training")
                assessed_labels = [assessor.get_rel_label(doc_id) for doc_id in selected_dids]
                assesed_features = ranker.get_feature_by_did(selected_dids)
                await ranker.train(assesed_features, assessed_labels)
                await ranker.remove_docs_from_cache(selected_dids)
                LOGGER.info(f"Assessed document training complete - {len(selected_dids):,} documents")
            iteration_duration = datetime.now() - iteration_start_time
            experiment.checkpoint(step=t, primary_metric=("running_true_recall", "maximize"), metrics={"run_group": param_group_name, "metric_type": MetricType.step.name,
                "iteration": t,
"batch_size": batch_size,
"total_num": total_num,
"sampled_num": sampled_num,
"total_true_r": total_true_r,
"running_true_r": running_true_r,
"ap": ap,
"running_true_recall": running_true_recall,
"sampled_percentage": sampled_percentage,
"iteration_duration_seconds":iteration_duration.total_seconds(),
"iteration_duration":str(iteration_duration)})
    
    stop_time = datetime.now()
    shown_dids = assessor.get_assessed_dids()
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)

    elapsed_run_time = stop_time-start_time
    final_metrics = eval(tar_run_file, qrel_file)
    experiment.checkpoint(path=os.path.relpath(tar_run_file), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.final.name, "calculated_metrics": final_metrics, "elapsed_time": str(elapsed_run_time), "elapsed_seconds": elapsed_run_time.total_seconds(), "nodes": ranker.model.weight_ab.shape[0]})
    LOGGER.info(f'TAR is finished. Elapsed: {elapsed_run_time}')

    return

def build_experiments(model_name, corpus_name, topics, vectorizer_types, run_notes = None, run_group = None):
    build_timestamp = datetime.now().isoformat().replace("-", "_")
    run_grouping = ""
    if run_group:
        run_grouping = f"{run_group}-"
    built_experiments = {}
    for topic in topics:
        for vectorizer_type in vectorizer_types:
            experiment_name = f"{run_grouping}{model_name}-{corpus_name}-{topic}-{vectorizer_type.name}-{build_timestamp}"
            built_experiments[experiment_name] = {"corpus_params": {"corpus_name": corpus_name, "collection_name": corpus_name, "topic_id": topic, "topic_set": topic}, "vectorizer_params": None, "vectorizer_type": vectorizer_type, "run_notes": run_notes}

    return built_experiments

async def main():
    await fuzzy_artmap_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"], classifier_params=fuzzy_artmap_params)

if __name__ == '__main__':

    tf_idf_params = make_tf_idf_vectorizer_params(0.001, 0.9, 'english')
    # vectorizer_type = VectorizerType.tf_idf
    x = tf_idf_params

    
    fuzzy_artmap_params = make_fuzzy_artmap_params(0.95, 50, "famdg", False)
    fuzzy_artmap_params["scheduler_address"] = "localhost:8786"

    # experiments = build_experiments(fuzzy_artmap_params["model_type"], "20newsgroups", ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"], [VectorizerType.tf_idf, VectorizerType.glove, VectorizerType.sbert, VectorizerType.word2vec], "re-run with fixed training labels")
    # experiments = build_experiments(fuzzy_artmap_params["model_type"], "reuters21578", ["earn", "money-fx", "crude"], [VectorizerType.tf_idf, VectorizerType.glove, VectorizerType.sbert, VectorizerType.word2vec], "re-run with fixed training labels")
    # experiments = {"famg-20newsgroups-forsale-tf_idf-local_perf-rerun": {"corpus_params": {"corpus_name": "20newsgroups", "collection_name": "20newsgroups", "topic_id": "misc.forsale", "topic_set": "misc.forsale"}, "vectorizer_params": None, "vectorizer_type": VectorizerType.tf_idf, "run_notes": "local perf test FAMG CPU re-run"}}
    # experiments = {"famg-reuters21578-earn-tf_idf-quick_test": {"corpus_params": {"corpus_name": "reuters21578", "collection_name": "reuters21578", "topic_id": "earn", "topic_set": "misc.forsale"}, "vectorizer_params": None, "vectorizer_type": VectorizerType.tf_idf, "run_notes": "quick test max nodes mode"}}
    experiments = build_experiments(fuzzy_artmap_params["model_type"], "reuters21578", ["earn"], [VectorizerType.tf_idf], "testing GPU & GC perf mods")
    # experiments = build_experiments(fuzzy_artmap_params["model_type"], "reuters21578", ["crude"], [VectorizerType.tf_idf], "testing GPU & GC perf mods")
    for param_group_name, experiment_params in experiments.items():
        LOGGER.info(f"starting experiment: {param_group_name}")
        corpus_params = make_file_params(**experiment_params["corpus_params"])
        
        experiment = keepsake.init(params={"run_group": param_group_name, "metric_type": MetricType.init.name, "corpus_params": corpus_params, "vectorizer_type": experiment_params["vectorizer_type"].name, "vectorizer_params": experiment_params["vectorizer_params"], "classifier_params": fuzzy_artmap_params, "random_state": json.dumps(random.getstate()), "run_notes": experiment_params["run_notes"]})
        # if fuzzy_artmap_params["model_type"] != "famdg":
        #     fuzzy_artmap_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"], classifier_params=fuzzy_artmap_params)
        # else:
        tornado.ioloop.IOLoop.current().run_sync(main)
        # tornado.ioloop.IOLoop.instance().start()
        LOGGER.info(f"experiment complete: {param_group_name}")
        gc.collect()
        gc.collect()
        # fuzzy_artmap_method(data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file)
