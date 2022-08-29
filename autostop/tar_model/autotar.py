# coding=utf-8
"""
The implementation is based on the following paper:
[1] Gordon V. Cormack and Maura R. Grossman. 2015. Autonomy and Reliability of Continuous Active Learning for
Technology-Assisted Review. CoRR abs/1504.06868 (2015). arXiv:1504.06868 http://arxiv.org/abs/1504.06868

"""
import json
import random
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/measures'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/trec_eval/seeker'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/tar_framework'))
print(sys.path)
import csv
import math
from datetime import datetime

import traceback
import gc
import subprocess
import argparse

import keepsake
import numpy as np
from operator import itemgetter

from tar_framework.assessing import DataLoader, Assessor
from tar_framework.ranking import Ranker, VectorizerType
from tar_model.utils import *
from tar_framework.utils import *

from trec_eval.tar_eval import main as eval

def get_traceback_string(e: Exception):
    if e is None:
        return "Passed exception is none!"
    if sys.version_info.minor >= 10:
        return ''.join(traceback.format_exception(e))
    else:
        return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


def autotar_method(data_name, topic_set, topic_id,
                   query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                   stopping_percentage=1.0, stopping_recall=None,  # autostop parameters, for debug
                   ranker_tfidf_corpus_files=[], classifier='lr', min_df=2, C=1.0,  # ranker parameters
                   random_state=0,
                vectorizer_params=None,
                vectorizer_type=VectorizerType.tf_idf,
                classifier_params=None,
                **kwargs):
    """
    Implementation of the TAR process.
    @param data_name: dataset name
    @param topic_set: parameter-tuning set or test set
    @param topic_id: topic id
    @param stopping_percentage: stop TAR when x percentage of documents have been screened
    @param stopping_recall: stop TAR when x recall is achieved
    @param corpus_type: indicates what corpus to use when building features, see Ranker
    @param min_df: parameter of Ranker
    @param C: parameter of Ranker
    @param save_did2feature: save the did2feature dict as a pickle to fasten experiments
    @param random_state: random seed
    @return:
    """
    np.random.seed(random_state)

    # model named with its configuration
    model_name = 'autotar' + '-'
    model_name += 'sp' + str(stopping_percentage) + '-'
    model_name += 'sr' + str(stopping_recall) + '-'
    model_name += 'ct' + str(ranker_tfidf_corpus_files) + '-'
    model_name += 'csf' + classifier + '-'
    model_name += 'md' + str(min_df) + '-'
    model_name += 'c' + str(C)
    LOGGER.info('Model configuration: {}.'.format(model_name))

    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_dids()
    complete_pseudo_dids = assessor.get_complete_pseudo_dids()
    complete_pseudo_texts = assessor.get_complete_pseudo_texts()

    if ranker_tfidf_corpus_files == []:
        corpus_texts = assessor.get_complete_pseudo_texts()
    else:  # this branch is only available for autotar, to study the effect of ranker
        def read_temp(files):
            for file in files:
                for text in DataLoader.read_doc_texts_2_list(file):
                    yield text

        corpus_texts = read_temp(ranker_tfidf_corpus_files)
    did2label = assessor.get_did2label()
    total_true_r = assessor.get_total_rel_num()
    total_num = assessor.get_total_doc_num()

    # preparing document features
    ranker = Ranker(model_type=classifier, random_state=random_state, min_df=min_df, C=C)
    # ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=corpus_texts)
    ranker.set_did_2_feature(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=corpus_texts, vectorizer_type=vectorizer_type, corpus_name=data_name, vectorizer_params=vectorizer_params)
    ranker.set_features_by_name('complete_dids', complete_dids)

    start_time = datetime.now()

    # local parameters are set according to [1]
    stopping = False
    t = 0
    batch_size = 1
    temp_doc_num = 100

    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
    with open(interaction_file, 'w', encoding='utf8') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(("iteration", "batch_size", "total_num", "sampled_num", "total_true_r", "running_true_r", "ap", "running_true_recall"))
        while not stopping:
            iteration_start_time = datetime.now()
            t += 1            
            LOGGER.info(f'TAR: iteration={t}')

            train_dids, train_labels = assessor.get_training_data(temp_doc_num)
            train_features = ranker.get_feature_by_did(train_dids)
            ranker.train_sync(train_features, train_labels)

            test_features = ranker.get_features_by_name('complete_dids')
            scores = ranker.predict(test_features)

            zipped = sorted(zip(complete_dids, scores), key=itemgetter(1), reverse=True)
            ranked_dids, scores = zip(*zipped)

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
            batch_size += math.ceil(batch_size / 10)

            # debug: writing values
            iteration_duration = datetime.now() - iteration_start_time
            csvwriter.writerow((t, batch_size, total_num, sampled_num, total_true_r, running_true_r, ap, running_true_recall))
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
            # debug: stop early
            if stopping_recall:
                if running_true_recall >= stopping_recall:
                    stopping = True
            if stopping_percentage:
                if sampled_num >= int(total_num * stopping_percentage):
                    stopping = True

    stop_time = datetime.now()
    # tar run file
    shown_dids = assessor.get_assessed_dids()
    check_func = assessor.assess_state_check_func()
    tar_run_file = name_tar_run_file(data_name=data_name, model_name=model_name, topic_set=topic_set, exp_id=random_state, topic_id=topic_id)
    LOGGER.info(f'writing results to: {tar_run_file}')
    with open(tar_run_file, 'w', encoding='utf8') as f:
        write_tar_run_file(f=f, topic_id=topic_id, check_func=check_func, shown_dids=shown_dids)
    
    elapsed_run_time = stop_time-start_time
    final_metrics = eval(tar_run_file, qrel_file)
    experiment.checkpoint(path=os.path.relpath(tar_run_file), 
                          primary_metric=(None, None), 
                          metrics={"run_group": param_group_name, 
                                   "metric_type": MetricType.final.name, 
                                   "calculated_metrics": final_metrics, 
                                   "elapsed_time": str(elapsed_run_time), 
                                   "elapsed_seconds": elapsed_run_time.total_seconds()})

    LOGGER.info(f'TAR is finished. Elapsed: {elapsed_run_time}. r - {running_true_recall}')
    return


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def as_enum(d):
    if "__enum__" in d:
        _, member = d["__enum__"].split(".")
        return getattr(VectorizerType, member)
    elif "__np__" in d:
        return getattr(np, d["__np__"])
    else:
        return d

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        if isinstance(obj, type):
            return {"__np__":str(obj).split(".")[1].rstrip("'>")}

        return json.JSONEncoder.default(self, obj)

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
        param_group_name = param_group_name.replace("famdg", "autotar")
        LOGGER.info(f"starting experiment: {param_group_name} - ({experiment_counter}/{number_of_experiments})")
        corpus_params = make_file_params(**experiment_params["corpus_params"])
        
        experiment = keepsake.init(params={ "run_group": param_group_name, 
                                            "metric_type": MetricType.init.name, 
                                            "corpus_params": corpus_params, 
                                            "vectorizer_type": experiment_params["vectorizer_type"].name, 
                                            "vectorizer_params": json.dumps(experiment_params["vectorizer_params"], cls=EnumEncoder), 
                                            "classifier_params": fuzzy_artmap_params, 
                                            "random_state": json.dumps(random.getstate()), 
                                            "run_notes": experiment_params["run_notes"],
                                            "git_revision_hash": git_revision_hash,
                                            "git_short_hash": git_revision_short_hash
                                           })

        try:
            autotar_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"])
        except Exception as e:
            trace_back_string = get_traceback_string(e)
            LOGGER.error(f"Error <{e}>\ntraceback: {trace_back_string}\nrunning experiment {param_group_name}\ncontinuing to next experiment...")
        # tornado.ioloop.IOLoop.instance().start()
        LOGGER.info(f"experiment complete: {param_group_name}")
        gc.collect()
        gc.collect()
