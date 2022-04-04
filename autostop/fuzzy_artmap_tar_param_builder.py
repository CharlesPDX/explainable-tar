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

def make_fuzzy_artmap_params(vigilance, number_of_mapping_nodes, model_type, max_nodes = None):
    fuzzy_artmap_params ={"rho_a_bar": vigilance, "number_of_mapping_nodes": number_of_mapping_nodes, "model_type": model_type, "max_nodes": max_nodes}
    return fuzzy_artmap_params

def build_experiments(model_name, corpus_name, topics, vectorizer_types, run_notes = None, run_group = None):
    build_timestamp = datetime.now().isoformat().replace("-", "_")
    run_grouping = ""
    if run_group:
        run_grouping = f"{run_group}-"
    built_experiments = {}
    for vectorizer_type in vectorizer_types:
        for topic in topics:
            experiment_name = f"{run_grouping}{model_name}-{corpus_name}-{topic}-{vectorizer_type.name}-{build_timestamp}"
            built_experiments[experiment_name] = {"corpus_params": {"corpus_name": corpus_name, "collection_name": corpus_name, "topic_id": topic, "topic_set": topic}, "vectorizer_params": None, "vectorizer_type": vectorizer_type, "run_notes": run_notes}

    return built_experiments

fuzzy_artmap_params = make_fuzzy_artmap_params(0.95, 50, "famdg")
fuzzy_artmap_params["scheduler_address"] = "localhost:8786"

experiments = build_experiments(fuzzy_artmap_params["model_type"], "20newsgroups", ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"], [VectorizerType.tf_idf, VectorizerType.glove, VectorizerType.sbert, VectorizerType.word2vec], "perf-test run")
experiments.update(build_experiments(fuzzy_artmap_params["model_type"], "reuters21578", ["earn", "money-fx", "crude"], [VectorizerType.tf_idf, VectorizerType.glove, VectorizerType.sbert, VectorizerType.word2vec], "perf-test run"))

destination_path = os.path.join(os.path.dirname(__file__), "tar_model")
params_file_name = "params.json"
params_file_location = os.path.join(destination_path, params_file_name)
with open(params_file_location, "w") as params_file:
    json.dump({"fuzzy_artmap_params": fuzzy_artmap_params, "experiments": experiments}, params_file, indent=4, cls=EnumEncoder)
