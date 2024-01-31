# coding=utf-8
from datetime import datetime
import sys
import os

import traceback
import gc
import subprocess
import argparse
from enum import Enum

import json
import csv
import random
import numpy as np
from operator import itemgetter

import keepsake
import tornado.ioloop

from tar_framework.assessing import Assessor
from tar_framework.ranking import Ranker, VectorizerType
from tar_framework.fuzzy_artmap_distributed_gpu import ProcessingMode
from metric_utilities import MetricType, calculate_ap
from parameter_utilities import make_file_params
from tar_framework.run_utilities import LOGGER, name_tar_run_file, write_tar_run_file, name_interaction_file, REL

from trec_eval.tar_eval import main as eval

# import cProfile
# import pstats
# profiler = cProfile.Profile()

def get_traceback_string(e: Exception):
    if e is None:
        return "Passed exception is none!"
    if sys.version_info.minor >= 10:
        return ''.join(traceback.format_exception(e))
    else:
        return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


async def retrain_model(ranker: Ranker, assessor: Assessor) -> None:
    relevant_document_ids = assessor.get_assessed_relevant_document_ids()
    positive_doc_ids = random.sample(relevant_document_ids, len(relevant_document_ids))
    assert len(positive_doc_ids) == len(relevant_document_ids)
    retraining_labels = list(len(positive_doc_ids) * [1])
    retraining_features = ranker.get_feature_by_document_ids(positive_doc_ids)
    LOGGER.info("Re-Caching corpus")
    ranker.model = None
    await ranker.cache_corpus_in_model(assessor.get_complete_document_ids()) # TODO: test removing already evaluated IDs
    LOGGER.info("Re-Caching complete")
    LOGGER.info("Begining retraining")
    await ranker.retrain_model(retraining_features, retraining_labels, positive_doc_ids)
    

async def fuzzy_artmap_method(data_name, topic_set, topic_id,
                query_file, qrel_file, doc_id_file, doc_text_file,  # data parameters
                stopping_percentage=1.0, stopping_recall=None,  # autostop parameters
                random_state=0,
                vectorizer_params=None,
                vectorizer_type=VectorizerType.tf_idf,
                classifier_params=None,
                retrain_count=0,
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
    model_name = f"fam-sp{str(stopping_percentage)}-sr-{str(stopping_recall)}"
    
    LOGGER.info(f"Model configuration: {model_name}.")
    
    # profiler.enable()
    # loading data
    assessor = Assessor(query_file, qrel_file, doc_id_file, doc_text_file)
    complete_dids = assessor.get_complete_document_ids()
    complete_pseudo_dids = assessor.get_complete_document_ids_with_pseudo_document()
    complete_pseudo_texts = assessor.get_complete_document_texts_with_pseudo_document()
    did2label = assessor.get_document_id_to_label()
    total_true_r = assessor.get_relevant_document_count()
    total_num = assessor.get_document_count()

    # local parameters
    stopping = False
    t = 0
    batch_size = 100
    was_retrained = False
    number_of_retrainings = 0
    retrained_iteration = []
    node_count_at_retrain = 0
    number_of_increases_at_retrain = 0

    if "batch_size" not in classifier_params:
        classifier_params["batch_size"] = batch_size

    if "mode" not in classifier_params:
        classifier_params["mode"] = ProcessingMode.local

    # preparing document features
    ranker = Ranker(**classifier_params)
    ranker.set_document_ids_to_features(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts, vectorizer_type=vectorizer_type, corpus_name=data_name, vectorizer_params=vectorizer_params)
    ranker.set_features_by_name("complete_dids", complete_dids)
    LOGGER.info("Caching corpus")
    await ranker.cache_corpus_in_model(complete_dids)
    LOGGER.info("Caching complete")
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    
    # starting the TAR process
    start_time = datetime.now()
    # perform initial model training, with some positive examples
    shuffled_doc_ids = random.sample(assessor.get_complete_document_ids(), len(assessor.get_complete_document_ids()))
    initial_positive_doc_ids = list(filter(lambda doc_id: assessor.document_id_to_label[doc_id] == REL, shuffled_doc_ids))[:10]
    initial_negative_doc_ids = list(filter(lambda doc_id: assessor.document_id_to_label[doc_id] != REL, shuffled_doc_ids))[:90]
    initial_training_doc_ids = list(initial_positive_doc_ids)
    initial_training_doc_ids.extend(initial_negative_doc_ids)
    initial_training_labels = list(len(initial_positive_doc_ids) * [1])
    initial_training_labels.extend(len(initial_negative_doc_ids) * [0])
    initial_training_features = ranker.get_feature_by_document_ids(initial_training_doc_ids)
    
    LOGGER.info(f"starting initial training")
    await ranker.train(initial_training_features, initial_training_labels, initial_training_doc_ids)
    LOGGER.info(f"initial training complete - {len(initial_training_doc_ids):,} documents")

    last_r = 0
    interaction_file = name_interaction_file(data_name=data_name, model_name=model_name, topic_set=topic_set,
                                             exp_id=random_state, topic_id=topic_id)
    def write_results():
        shown_dids = assessor.get_assessed_document_ids()
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

            unassessed_document_ids = assessor.get_unassessed_document_ids()
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
            else:
                zipped = list(scores)
            
            if len(zipped) > 0:
                _, ranked_dids, _ = zip(*zipped)
            else:
                ranked_dids = []
            
            # cutting off instead of sampling
            selected_dids = assessor.get_top_assessed_document_ids(ranked_dids, batch_size)
            assessor.update_assessment(selected_dids)

            # statistics
            sampled_num = assessor.get_assessed_count()
            sampled_percentage = sampled_num/total_num
            running_true_r = assessor.get_assessed_relevant_count()
            running_true_recall = 0
            if total_true_r != 0:
                running_true_recall = running_true_r / float(total_true_r)
            ap = calculate_ap(did2label, ranked_dids)

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
                    if retrain_count > 0 and (number_of_retrainings + 1) <= retrain_count:
                        LOGGER.info("Retraining model")
                        number_of_retrainings += 1
                        was_retrained = True
                        retrained_iteration.append(t)
                        node_count_at_retrain += ranker.model.get_number_of_nodes()
                        number_of_increases_at_retrain += ranker.model.get_number_of_increases()
                        await retrain_model(ranker, assessor)
                        LOGGER.info("Retraining complete")
                    else:
                        LOGGER.info("Stopping")
                        stopping = True

            last_r = running_true_r

            # train model with new assessments
            if not stopping and not was_retrained:
                LOGGER.info("Starting assessed document training")
                assessed_labels = [assessor.get_rel_label(doc_id) for doc_id in selected_dids]
                assesed_features = ranker.get_feature_by_document_ids(selected_dids)
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
                "iteration_duration":str(iteration_duration),
                "was_retrained": was_retrained})
            was_retrained = False
            LOGGER.info(f"r: {running_true_recall}, ap: {ap}, sampled: {sampled_percentage}% - retrainings: {number_of_retrainings}")
    
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
                                   "nodes": ranker.model.get_number_of_nodes() + node_count_at_retrain,
                                   "number_of_increases": ranker.model.get_number_of_increases() + number_of_increases_at_retrain,
                                   "increase_size": ranker.model.get_increase_size(),
                                   "committed_nodes": ranker.model.get_committed_nodes(),
                                   "retrained_iteration": ",".join([str(interation) for interation in retrained_iteration])})

    LOGGER.info(f'TAR is finished. Elapsed: {elapsed_run_time}. r - {running_true_recall}')
    return


async def main():
    await fuzzy_artmap_method(**corpus_params, vectorizer_type=experiment_params["vectorizer_type"], vectorizer_params=experiment_params["vectorizer_params"], classifier_params=fuzzy_artmap_params, retrain_count=fuzzy_artmap_params.get("retrain_count", 0))

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

    params_file_location = os.path.join(os.getcwd(), "explainable-tar", args.params)
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
                                            "vectorizer_params": json.dumps(experiment_params["vectorizer_params"], cls=EnumEncoder), 
                                            "classifier_params": fuzzy_artmap_params, 
                                            "random_state": json.dumps(random.getstate()), 
                                            "run_notes": experiment_params["run_notes"],
                                            "git_revision_hash": git_revision_hash,
                                            "git_short_hash": git_revision_short_hash,
                                            "retrain_count": fuzzy_artmap_params.get("retrain_count", 0)
                                           })
        try:
            tornado.ioloop.IOLoop.current().run_sync(main)
        except tornado.iostream.StreamClosedError as stream_error:
            trace_back_string = get_traceback_string(stream_error)
            LOGGER.fatal(f"At least one worker is terminated, cannot continue, exiting on {param_group_name}\ntraceback: {trace_back_string}")
            raise
        except Exception as e:
            trace_back_string = get_traceback_string(e)
            LOGGER.error(f"Error <{e}>\ntraceback: {trace_back_string}\nrunning experiment {param_group_name}\ncontinuing to next experiment...")
        LOGGER.info(f"experiment complete: {param_group_name}")
        gc.collect()
        gc.collect()
    run_completion_time = datetime.now()
    run_duration = run_completion_time - run_start_time
    LOGGER.info(f"run completed\nstarted at {run_start_time}\nended at: {run_completion_time}\nelapsed: {run_duration}")
