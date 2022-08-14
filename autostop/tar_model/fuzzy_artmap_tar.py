# coding=utf-8
# from asyncio.log import logger
from datetime import datetime
# from logging import Logger
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/measures'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/seeker'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/tar_framework'))
print(sys.path)

import traceback
import gc
import subprocess
import argparse

import json
import csv
# import math
import random
# import numpy as np
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

def get_traceback_string(e: Exception):
    if e is None:
        return "Passed exception is none!"
    return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))

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

    # local parameters
    stopping = False
    t = 0
    batch_size = 100

    classifier_params["batch_size"] = batch_size

    # preparing document features
    ranker = Ranker(**classifier_params)
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts, vectorizer_type=vectorizer_type, corpus_name=data_name, vectorizer_params=vectorizer_params)
    ranker.set_features_by_name('complete_dids', complete_dids)
    LOGGER.info("Caching corpus")
    await ranker.cache_corpus_in_model(complete_dids)
    LOGGER.info("Caching complete")
        
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
    await ranker.train(initial_training_features, initial_training_labels, initial_training_doc_ids)
    LOGGER.info(f"initial training complete - {len(initial_training_doc_ids):,} documents")

    last_r = 0
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    def write_results():
        shown_dids = assessor.get_assessed_dids()
        check_func = assessor.assess_state_check_func()
        tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
        with open(tar_run_file, 'w', encoding='utf8') as f:
            write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)
        return tar_run_file
    
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(("iteration", "batch_size", "total_num", "sampled_num", "total_true_r", "running_true_r", "ap", "running_true_recall", "sampled_percentage"))
        gc.disable()
        while not stopping:
            iteration_start_time = datetime.now()
            t += 1
            LOGGER.info(f'TAR: iteration={t}')

            unassessed_document_ids = assessor.get_unassessed_dids()
            # test_features = ranker.get_feature_by_did(unassessed_document_ids)
            # scores = ranker.predict_with_doc_id(test_features, unassessed_document_ids)
            try:
                scores = await ranker.predict_with_doc_id(unassessed_document_ids)
            except Exception as e:
                tar_run_file = write_results()
                trace_back_string = get_traceback_string(e)
                model_path = ranker.save_model(param_group_name)
                experiment.checkpoint(path=os.path.relpath(model_path), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.model.name, "step": t})
                experiment.checkpoint(path=os.path.relpath(tar_run_file), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.error.name, "step": t, "error": repr(e), "traceback": trace_back_string})
                LOGGER.error(f"Error {e} - {trace_back_string} getting predictions\nresults so far saved to {tar_run_file}, model state saved to {model_path}")
                raise
            if classifier_params["active_learning_mode"] == "ranked":
                zipped = sorted(scores, key=itemgetter(0), reverse=True)
                if len(zipped) > 0:
                    _, ranked_dids = zip(*zipped)
                else:
                    ranked_dids = []
            else:
                _, ranked_dids = zip(*scores)

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
                try:
                    await ranker.train(assesed_features, assessed_labels, selected_dids)
                    await ranker.remove_docs_from_cache(selected_dids)
                except Exception as e:
                    trace_back_string = get_traceback_string(e)
                    tar_run_file = write_results()
                    model_path = ranker.save_model(param_group_name)
                    experiment.checkpoint(path=os.path.relpath(model_path), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.model.name, "step": t})
                    experiment.checkpoint(path=os.path.relpath(tar_run_file), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.error.name, "step": t, "error": repr(e), "traceback": trace_back_string})
                    LOGGER.error(f"Error {e} - {trace_back_string} training on updated docs\nresults so far saved to {tar_run_file}, model state saved to {model_path}")
                    raise
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
    tar_run_file = write_results()
    model_path = ranker.save_model(param_group_name)
    experiment.checkpoint(path=os.path.relpath(model_path), primary_metric=(None, None), metrics={"run_group": param_group_name, "metric_type": MetricType.model.name, "step": t})

    elapsed_run_time = stop_time-start_time
    final_metrics = eval(tar_run_file, qrel_file)
    experiment.checkpoint(path=os.path.relpath(tar_run_file), 
                          primary_metric=(None, None), 
                          metrics={"run_group": param_group_name, 
                                   "metric_type": MetricType.final.name, 
                                   "calculated_metrics": final_metrics, 
                                   "elapsed_time": str(elapsed_run_time), 
                                   "elapsed_seconds": elapsed_run_time.total_seconds(), 
                                   "nodes": ranker.model.get_number_of_nodes(),
                                   "number_of_increases": ranker.model.get_number_of_increases(),
                                   "increase_size": ranker.model.get_increase_size(),
                                   "committed_nodes": ranker.model.get_commited_nodes()})

    LOGGER.info(f'TAR is finished. Elapsed: {elapsed_run_time}. r - {running_true_recall}')
    return


async def main():
    await fuzzy_artmap_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"], classifier_params=fuzzy_artmap_params)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def as_enum(d):
    if "__enum__" in d:
        _, member = d["__enum__"].split(".")
        return getattr(VectorizerType, member)
    else:
        return d


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-p", "--params", help="file name of experiment params", required=True)
    args = arg_parser.parse_args()

    try:
        git_revision_hash = get_git_revision_hash()
        git_revision_short_hash = git_revision_hash[:7]
    except Exception as e:
        print(f"Exception retrieving git commit hash")
        git_revision_hash = "unavailable"
        git_revision_short_hash = "unavailable"

    # tf_idf_params = make_tf_idf_vectorizer_params(0.001, 0.9, 'english')
    # vectorizer_type = VectorizerType.tf_idf
    # x = tf_idf_params
    params_file_location = os.path.join(os.getcwd(), "autostop/tar_model", args.params)
    with open(params_file_location) as params_file:
        params = json.load(params_file, object_hook=as_enum)
        experiments = params["experiments"]

    run_start_time = datetime.now()
    number_of_experiments = len(experiments)
    LOGGER.info(f"Running {number_of_experiments} experiments")
    experiment_counter = 0
    for param_group_name, experiment_params in experiments.items():
        fuzzy_artmap_params = experiment_params["fuzzy_artmap_params"]
        experiment_counter += 1
        LOGGER.info(f"starting experiment: {param_group_name} - ({experiment_counter}/{number_of_experiments})")
        corpus_params = make_file_params(**experiment_params["corpus_params"])
        
        experiment = keepsake.init(params={ "run_group": param_group_name, 
                                            "metric_type": MetricType.init.name, 
                                            "corpus_params": corpus_params, 
                                            "vectorizer_type": experiment_params["vectorizer_type"].name, 
                                            "vectorizer_params": experiment_params["vectorizer_params"], 
                                            "classifier_params": fuzzy_artmap_params, 
                                            "random_state": json.dumps(random.getstate()), 
                                            "run_notes": experiment_params["run_notes"],
                                            "git_revision_hash": git_revision_hash,
                                            "git_short_hash": git_revision_short_hash
                                           })
        # if fuzzy_artmap_params["model_type"] != "famdg":
        #     fuzzy_artmap_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"], classifier_params=fuzzy_artmap_params)
        # else:
        try:
            tornado.ioloop.IOLoop.current().run_sync(main)
        except Exception as e:
            trace_back_string = get_traceback_string(e)
            LOGGER.error(f"Error <{e}>\ntraceback: {trace_back_string}\nrunning experiment {param_group_name}\ncontinuing to next experiment...")
        # tornado.ioloop.IOLoop.instance().start()
        LOGGER.info(f"experiment complete: {param_group_name}")
        gc.collect()
        gc.collect()
        # fuzzy_artmap_method(data_name, topic_id, topic_set, query_file, qrel_file, doc_id_file, doc_text_file)
    run_completion_time = datetime.now()
    run_duration = run_completion_time - run_start_time
    LOGGER.info(f"run completed\nstarted at {run_start_time}\nended at: {run_completion_time}\nelapsed: {run_duration}")