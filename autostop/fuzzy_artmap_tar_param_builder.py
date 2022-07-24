from enum import Enum
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/tar_framework'))

from datetime import datetime
import json

from autostop.tar_framework.ranking import VectorizerType

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)

def make_fuzzy_artmap_params(vigilance, number_of_mapping_nodes, model_type, max_nodes = None, committed_beta = 0.75, active_learning_mode = "ranked"):
    fuzzy_artmap_params ={"rho_a_bar": vigilance, "number_of_mapping_nodes": number_of_mapping_nodes, "model_type": model_type, "max_nodes": max_nodes, "committed_beta": committed_beta, "active_learning_mode": active_learning_mode}
    return fuzzy_artmap_params

def build_experiments(fuzzy_artmap_params, corpus_name, topics, vectorizer_types, run_notes = None, run_group = None):
    model_name = fuzzy_artmap_params["model_type"]
    build_timestamp = datetime.now().isoformat().replace("-", "_")
    run_grouping = ""
    if run_group:
        run_grouping = f"{run_group}-"
    built_experiments = {}
    for vectorizer_type in vectorizer_types:
        for topic in topics:
            experiment_name = f"{run_grouping}{model_name}-{corpus_name}-{topic}-{vectorizer_type.name}-{build_timestamp}"
            built_experiments[experiment_name] = {"corpus_params": {"corpus_name": corpus_name, "collection_name": corpus_name, "topic_id": topic, "topic_set": topic}, 
                                                  "vectorizer_params": None, 
                                                  "vectorizer_type": vectorizer_type, 
                                                  "run_notes": run_notes,
                                                  "fuzzy_artmap_params": fuzzy_artmap_params}

    return built_experiments

def build_experiment_for_corpus_and_topics(corpus_and_topics, fuzzy_artmap_params, vectorizer_types, experiments, run_notes):
    for corpus, topics in corpus_and_topics.items():
        experiments.update(build_experiments(fuzzy_artmap_params, corpus, topics, vectorizer_types, run_notes))

def param_builder(filename_prefix, rho, beta, corpus_and_topics, run_notes, active_learning_mode):
    default_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, 50, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    default_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    default_vectorizer_types = [VectorizerType.tf_idf, VectorizerType.glove]

    word2vec_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, 3200, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    word2vec_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    word2vec_vectorizer_types = [VectorizerType.word2vec]

    sbert_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, 6500, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    sbert_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    sbert_vectorizer_types = [VectorizerType.sbert]

    experiments = {}    
    build_experiment_for_corpus_and_topics(corpus_and_topics, default_fuzzy_artmap_params, default_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, word2vec_fuzzy_artmap_params, word2vec_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, sbert_fuzzy_artmap_params, sbert_vectorizer_types, experiments, run_notes)

    destination_path = os.path.join(os.path.dirname(__file__), "tar_model")
    params_file_name = f"{filename_prefix}_params.json"
    params_file_location = os.path.join(destination_path, params_file_name)
    with open(params_file_location, "w") as params_file:
        json.dump({"experiments": experiments}, params_file, indent=4, cls=EnumEncoder)

# rho = 0.6
# beta = 0.75
# reuters_small_test ={
#     # "20newsgroups": ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"],
#     "reuters21578": ["earn", "money-fx", "crude"]
#     # "reuters21578": ["earn"],
#     # "20newsgroups": ["comp.sys.ibm.pc.hardware"],
# }

learning_rate_run ={
    # "20newsgroups": ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"],
    # "reuters21578": ["earn", "money-fx", "crude"]
    "reuters21578": ["earn"],
    # "20newsgroups": ["comp.sys.ibm.pc.hardware"],
}
active_learning_mode_for_experiments = "ranked" # or "random"
experiments = [
    # ("lemma_alpha", 0.95, 0.80, learning_rate_run, "vigilance .95 with slow recode beta 0.80 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_beta", 0.95, 0.85, learning_rate_run, "vigilance .95 with slow recode beta 0.85 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_gamma", 0.95, 0.90, learning_rate_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_delta", 0.95, 0.95, learning_rate_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    ("debug_run", 0.95, 1.0, learning_rate_run, "vigilance .95 with fast learn beta 1.0 Reuters, debug run", active_learning_mode_for_experiments)
]

for experiment in experiments:
    param_builder(*experiment)
