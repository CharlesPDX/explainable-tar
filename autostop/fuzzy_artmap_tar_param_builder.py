from enum import Enum
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop'))
sys.path.insert(0, os.path.join(os.getcwd(), 'autostop/tar_framework'))

from datetime import datetime
import json

import numpy as np
import jsonpickle

from autostop.tar_framework.ranking import VectorizerType

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        if isinstance(obj, type):
            return {"__np__":str(obj).split(".")[1].rstrip("'>")}

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
        vectorizer_params = None
        if vectorizer_type == VectorizerType.tf_idf and ("athome" in corpus_name or "rcv1" in corpus_name):
            vectorizer_params = {'stop_words': 'english', 'min_df': 2, 'dtype': np.float32}
        for topic in topics:
            collection_name = corpus_name
            if "athome" in corpus_name:
                collection_name = "tr"

            experiment_name = f"{run_grouping}{model_name}-{corpus_name}-{topic}-{vectorizer_type.name}-{build_timestamp}"
            built_experiments[experiment_name] = {"corpus_params": {"corpus_name": corpus_name, "collection_name": collection_name, "topic_id": topic, "topic_set": topic}, 
                                                  "vectorizer_params": vectorizer_params, 
                                                  "vectorizer_type": vectorizer_type, 
                                                  "run_notes": run_notes,
                                                  "fuzzy_artmap_params": fuzzy_artmap_params}

    return built_experiments

def build_experiment_for_corpus_and_topics(corpus_and_topics, fuzzy_artmap_params, vectorizer_types, experiments, run_notes, vectorizer_params = None):
    for corpus, topics in corpus_and_topics.items():
        experiments.update(build_experiments(fuzzy_artmap_params, corpus, topics, vectorizer_types, run_notes, vectorizer_params))

def param_builder(filename_prefix, rho, beta, corpus_and_topics, run_notes, active_learning_mode, use_large_corpus = False):
    tf_idf_starting_nodes = 200
    glove_starting_nodes = 200
    word2vec_starting_nodes = 3200
    sbert_starting_nodes = 6500

    if use_large_corpus:
        tf_idf_starting_nodes = 50
        glove_starting_nodes = 20_000
        word2vec_starting_nodes = 40_000
        sbert_starting_nodes = 60_000    

    default_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, tf_idf_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    default_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    default_vectorizer_types = [VectorizerType.tf_idf]

    glove_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, glove_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    glove_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    glove_vectorizer_types = [VectorizerType.glove]

    word2vec_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, word2vec_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    word2vec_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    word2vec_vectorizer_types = [VectorizerType.word2vec]

    sbert_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, sbert_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode)
    sbert_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    sbert_vectorizer_types = [VectorizerType.sbert]

    experiments = {}    
    build_experiment_for_corpus_and_topics(corpus_and_topics, default_fuzzy_artmap_params, default_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, glove_fuzzy_artmap_params, glove_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, word2vec_fuzzy_artmap_params, word2vec_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, sbert_fuzzy_artmap_params, sbert_vectorizer_types, experiments, run_notes)

    destination_path = os.path.join(os.path.dirname(__file__), "tar_model")
    params_file_name = f"{filename_prefix}_params.json"
    params_file_location = os.path.join(destination_path, params_file_name)
    # jsonpickle.set_encoder_options('simplejson', encoding='utf8')
    # encoded = jsonpickle.encode({"experiments": experiments}, unpicklable=False, use_decimal=True, indent=4) # .encode('ascii','ignore')
    with open(params_file_location, "w") as params_file:
        json.dump({"experiments": experiments}, params_file, indent=4, cls=EnumEncoder)

reuters_small_test ={
    # "20newsgroups": ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"],
    # "reuters21578": ["earn", "money-fx", "crude"]
    "reuters21578": ["earn"],
    # "20newsgroups": ["comp.sys.ibm.pc.hardware"],
}

re_run ={
    "20newsgroups": ["comp.sys.ibm.pc.hardware", "sci.med", "misc.forsale"],
    "reuters21578": ["earn", "money-fx", "crude"]
}

newsgroups = ["sci.crypt", "misc.forsale", "sci.med", "rec.sport.hockey", "alt.atheism", "comp.sys.mac.hardware", "alt.atheism.short", "comp.os.ms-windows.misc", "talk.politics.mideast", "soc.religion.christian", "talk.politics.misc", "talk.politics.guns", "rec.motorcycles", "comp.windows.x", "comp.graphics", "rec.sport.baseball", "comp.sys.ibm.pc.hardware", "sci.electronics", "sci.space", "rec.autos", "talk.religion.misc"]
small_reuters_topics = ["oilseed", "rapeseed", "copra-cake", "nkr", "wool", "bfr", "jet", "barley", "tung", "coffee", "housing", "trade", "red-bean", "lupin", "rye", "citruspulp", "can", "stg", "soy-meal", "coconut-oil", "castor-meal", "sfr", "groundnut", "ffr", "flaxseed", "instal-debt", "austdlr", "singdlr", "hog", "copper", "saudriyal", "nat-gas", "dfl", "sugar", "mexpeso", "tapioca", "austral", "corn-oil", "plywood", "rice", "dmk", "heat", "sorghum", "soy-oil", "alum", "f-cattle", "jobs", "cottonseed", "corn", "lin-oil", "lead", "silver", "rape-meal", "lin-meal", "potato", "cruzado", "nzdlr", "income", "wpi", "drachma", "money-supply", "orange", "gas", "lei", "cpu", "groundnut-oil", "strategic-metal", "propane", "rand", "acq", "rupiah", "palladium", "iron-steel", "carcass", "rape-oil", "sunseed", "fishmeal", "oat", "silk", "cocoa", "castorseed", "cpi", "platinum", "palm-meal", "pet-chem", "skr", "pork-belly", "palm-oil", "dkr", "interest", "groundnut-meal", "rubber", "sun-meal", "tin", "dlr", "lumber", "veg-oil", "naphtha", "meal-feed", "lit", "cotton-oil", "grain", "crude", "inventories", "hk", "palmkernel", "gold", "ship", "soybean", "money-fx", "wheat", "fuel", "escudo", "ringgit", "ipi", "retail", "earn", "livestock", "cotton-meal", "yen", "l-cattle", "gnp", "coconut", "tea", "linseed", "bop", "peseta", "castor-oil", "cotton", "zinc", "tung-oil", "nickel", "cornglutenfeed", "reserves", "sun-oil"]

full_topics={
    "20newsgroups": ["sci.crypt", "misc.forsale", "sci.med", "rec.sport.hockey", "alt.atheism", "comp.sys.mac.hardware", "alt.atheism.short", "comp.os.ms-windows.misc", "talk.politics.mideast", "soc.religion.christian", "talk.politics.misc", "talk.politics.guns", "rec.motorcycles", "comp.windows.x", "comp.graphics", "rec.sport.baseball", "comp.sys.ibm.pc.hardware", "sci.electronics", "sci.space", "rec.autos", "talk.religion.misc"],
    "reuters21578": ["oilseed", "rapeseed", "copra-cake", "nkr", "wool", "bfr", "jet", "barley", "tung", "coffee", "housing", "trade", "red-bean", "lupin", "rye", "citruspulp", "can", "stg", "soy-meal", "coconut-oil", "castor-meal", "sfr", "groundnut", "ffr", "flaxseed", "instal-debt", "austdlr", "singdlr", "hog", "copper", "saudriyal", "nat-gas", "dfl", "sugar", "mexpeso", "tapioca", "austral", "corn-oil", "plywood", "rice", "dmk", "heat", "sorghum", "soy-oil", "alum", "f-cattle", "jobs", "cottonseed", "corn", "lin-oil", "lead", "silver", "rape-meal", "lin-meal", "potato", "cruzado", "nzdlr", "income", "wpi", "drachma", "money-supply", "orange", "gas", "lei", "cpu", "groundnut-oil", "strategic-metal", "propane", "rand", "acq", "rupiah", "palladium", "iron-steel", "carcass", "rape-oil", "sunseed", "fishmeal", "oat", "silk", "cocoa", "castorseed", "cpi", "platinum", "palm-meal", "pet-chem", "skr", "pork-belly", "palm-oil", "dkr", "interest", "groundnut-meal", "rubber", "sun-meal", "tin", "dlr", "lumber", "veg-oil", "naphtha", "meal-feed", "lit", "cotton-oil", "grain", "crude", "inventories", "hk", "palmkernel", "gold", "ship", "soybean", "money-fx", "wheat", "fuel", "escudo", "ringgit", "ipi", "retail", "earn", "livestock", "cotton-meal", "yen", "l-cattle", "gnp", "coconut", "tea", "linseed", "bop", "peseta", "castor-oil", "cotton", "zinc", "tung-oil", "nickel", "cornglutenfeed", "reserves", "sun-oil"],
    # "reuters-rcv1": ["M131", "8YDB", "GCRIM", "G153", "GDIP", "C312", "GCAT", "E31", "GSPO", "C313", "G13", "MEUR", "G155", "E312", "M142", "E61", "GFAS", "C17", "C16", "GVOTE", "G158", "E132", "GVIO", "C32", "C41", "M13", "E411", "C171", "G159", "E511", "E512", "C12", "GWELF", "9BNX", "E21", "G154", "1POL", "E14", "ADS10", "GENV", "E12", "GDIS", "C331", "GEDU", "G112", "G156", "CCAT", "C182", "G152", "GSCI", "C411", "C152", "M132", "G11", "C33", "PRB13", "E142", "G12", "C21", "M12", "MCAT", "GWEA", "C311", "C151", "GTOUR", "C174", "C11", "GMIL", "E131", "7RSK", "G113", "BRP11", "C31", "E13", "C172", "2ECO", "GHEA", "M11", "C42", "C1511", "G14", "E311", "C24", "6INS", "GPOL", "G111", "GJOB", "E51", "C13", "GOBIT", "G15", "C14", "G151", "C18", "GDEF", "E11", "E211", "G131", "BNW14", "C34", "GODD", "ENT12", "E143", "E41", "E212", "C181", "4GEN", "C173", "GREL", "GPRO", "C183", "GENT", "G157", "E71", "M14", "C23", "C22", "ECAT", "E141", "C15", "E121", "M143", "M141", "3SPO", "E313", "E513"],
    # "tr": ["433", "413", "athome102", "407", "athome104", "athome105", "410", "432", "416", "422", "414", "405", "434", "athome103", "athome107", "423", "403", "412", "409", "401", "athome101", "athome109", "426", "408", "402", "429", "418", "428", "athome106", "427", "419", "athome108", "430", "411", "415", "421", "417", "424", "404", "431", "425", "406", "athome100", "420"]
}

active_learning_mode_for_experiments = "random" #"ranked" # or "random"
# experiments = [
    # ("lemma_alpha", 0.95, 0.80, learning_rate_run, "vigilance .95 with slow recode beta 0.80 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_beta", 0.95, 0.85, learning_rate_run, "vigilance .95 with slow recode beta 0.85 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_gamma", 0.95, 0.90, learning_rate_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_delta", 0.95, 0.95, learning_rate_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("debug_run", 0.95, 1.0, learning_rate_run, "vigilance .95 with fast learn beta 1.0 Reuters, debug run", active_learning_mode_for_experiments)
# ]

# experiments = [
#     ("rerun_alpha", 0.60, 0.75, re_run, "vigilance .95 with slow recode beta 0.80 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_beta", 0.60, 1.0, re_run, "vigilance .95 with slow recode beta 0.85 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_gamma", 0.90, 0.75, re_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_delta", 0.90, 1.0, re_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_epsilon", 0.95, 0.75, re_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_zeta", 0.95, 1.0, re_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
# ]

rho = 0.95
beta = 1.0
experiments = []
chunk_size = 3
prefixes = []
newsgroups_prefix = "golf"
reuters_small_prefix = "hotel"
for chunk_index, i in enumerate(range(0, len(newsgroups), chunk_size)):
    topics = {"20newsgroups": newsgroups[i:i+chunk_size]}
    prefixes.append(f'"{newsgroups_prefix}_{chunk_index}"')
    experiments.append((f"{newsgroups_prefix}_{chunk_index}", rho, beta, topics, f"vigilance {rho} with beta {beta} 20Newsgroups topics - {active_learning_mode_for_experiments} active learning mode", active_learning_mode_for_experiments))

for chunk_index, i in enumerate(range(0, len(small_reuters_topics), chunk_size)):
    topics = {"reuters21578": small_reuters_topics[i:i+chunk_size]}
    prefixes.append(f'"{reuters_small_prefix}_{chunk_index}"')
    experiments.append((f"{reuters_small_prefix}_{chunk_index}", rho, beta, topics, f"vigilance {rho} with beta {beta} Reuters-21578 topics - {active_learning_mode_for_experiments} active learning mode", active_learning_mode_for_experiments))

# test_topics={
#     "reuters-rcv1": ["M131"],
#     # "tr": ["433", "413", "athome102", "407", "athome104", "athome105", "410", "432", "416", "422", "414", "405", "434", "athome103", "athome107", "423", "403", "412", "409", "401", "athome101", "athome109", "426", "408", "402", "429", "418", "428", "athome106", "427", "419", "athome108", "430", "411", "415", "421", "417", "424", "404", "431", "425", "406", "athome100", "420"]
# }

test_topics={
    "athome1": ["athome102"]
}

# test_topics={
#     "athome4": ["401"]
# }

# experiments = [
#     # ("test_rcv1", rho, beta, test_topics, f"vigilance {rho} with beta {beta}", active_learning_mode_for_experiments, True),
#     ("test_small_reuters", rho, beta, reuters_small_test, f"vigilance {rho} with beta {beta}", active_learning_mode_for_experiments, False),
# ]

for experiment in experiments:
    param_builder(*experiment)

combined_prefixes = ", ".join(prefixes)
print(f"[{combined_prefixes}]")