from enum import Enum
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), 'tar_framework'))

from datetime import datetime
import json

import numpy as np
import jsonpickle

from tar_framework.ranking import VectorizerType
from tar_framework.fuzzy_artmap_distributed_gpu import ProcessingMode

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        if isinstance(obj, type):
            return {"__np__":str(obj).split(".")[1].rstrip("'>")}

        return json.JSONEncoder.default(self, obj)

def make_fuzzy_artmap_params(vigilance, number_of_mapping_nodes, model_type, max_nodes = None, committed_beta = 0.75, active_learning_mode = "ranked", number_of_retrainings = 0, mode = ProcessingMode.distributed):
    fuzzy_artmap_params ={"rho_a_bar": vigilance, "number_of_mapping_nodes": number_of_mapping_nodes, "model_type": model_type, "max_nodes": max_nodes, "committed_beta": committed_beta, "active_learning_mode": active_learning_mode, "retrain_count": number_of_retrainings, "mode": mode}
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

def param_builder(filename_prefix, rho, beta, corpus_and_topics, run_notes, active_learning_mode, use_large_corpus = False, number_of_retrainings = 0, mode = ProcessingMode.distributed):
    tf_idf_starting_nodes = 200
    glove_starting_nodes = 3000
    word2vec_starting_nodes = 3200
    sbert_starting_nodes = 6500

    if use_large_corpus:
        tf_idf_starting_nodes = 50
        glove_starting_nodes = 12_000
        word2vec_starting_nodes = 12_000
        sbert_starting_nodes = 16_000

    default_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, tf_idf_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode, number_of_retrainings=number_of_retrainings)
    default_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    default_vectorizer_types = [VectorizerType.tf_idf]

    glove_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, glove_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode, number_of_retrainings=number_of_retrainings)
    glove_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    glove_vectorizer_types = [VectorizerType.glove]

    word2vec_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, word2vec_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode, number_of_retrainings=number_of_retrainings)
    word2vec_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    word2vec_vectorizer_types = [VectorizerType.word2vec]

    sbert_fuzzy_artmap_params = make_fuzzy_artmap_params(rho, sbert_starting_nodes, "famdg", committed_beta=beta, active_learning_mode=active_learning_mode, number_of_retrainings=number_of_retrainings)
    sbert_fuzzy_artmap_params["scheduler_address"] = "localhost:8786"
    sbert_vectorizer_types = [VectorizerType.sbert]

    experiments = {}    
    # build_experiment_for_corpus_and_topics(corpus_and_topics, default_fuzzy_artmap_params, default_vectorizer_types, experiments, run_notes)
    # build_experiment_for_corpus_and_topics(corpus_and_topics, glove_fuzzy_artmap_params, glove_vectorizer_types, experiments, run_notes)
    # build_experiment_for_corpus_and_topics(corpus_and_topics, word2vec_fuzzy_artmap_params, word2vec_vectorizer_types, experiments, run_notes)
    build_experiment_for_corpus_and_topics(corpus_and_topics, sbert_fuzzy_artmap_params, sbert_vectorizer_types, experiments, run_notes)

    destination_path = os.path.join(os.path.dirname(__file__), "")
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
    # "20newsgroups": ["sci.crypt", "misc.forsale", "sci.med", "rec.sport.hockey", "alt.atheism", "comp.sys.mac.hardware", "alt.atheism.short", "comp.os.ms-windows.misc", "talk.politics.mideast", "soc.religion.christian", "talk.politics.misc", "talk.politics.guns", "rec.motorcycles", "comp.windows.x", "comp.graphics", "rec.sport.baseball", "comp.sys.ibm.pc.hardware", "sci.electronics", "sci.space", "rec.autos", "talk.religion.misc"],
    # "reuters21578": ["oilseed", "rapeseed", "copra-cake", "nkr", "wool", "bfr", "jet", "barley", "tung", "coffee", "housing", "trade", "red-bean", "lupin", "rye", "citruspulp", "can", "stg", "soy-meal", "coconut-oil", "castor-meal", "sfr", "groundnut", "ffr", "flaxseed", "instal-debt", "austdlr", "singdlr", "hog", "copper", "saudriyal", "nat-gas", "dfl", "sugar", "mexpeso", "tapioca", "austral", "corn-oil", "plywood", "rice", "dmk", "heat", "sorghum", "soy-oil", "alum", "f-cattle", "jobs", "cottonseed", "corn", "lin-oil", "lead", "silver", "rape-meal", "lin-meal", "potato", "cruzado", "nzdlr", "income", "wpi", "drachma", "money-supply", "orange", "gas", "lei", "cpu", "groundnut-oil", "strategic-metal", "propane", "rand", "acq", "rupiah", "palladium", "iron-steel", "carcass", "rape-oil", "sunseed", "fishmeal", "oat", "silk", "cocoa", "castorseed", "cpi", "platinum", "palm-meal", "pet-chem", "skr", "pork-belly", "palm-oil", "dkr", "interest", "groundnut-meal", "rubber", "sun-meal", "tin", "dlr", "lumber", "veg-oil", "naphtha", "meal-feed", "lit", "cotton-oil", "grain", "crude", "inventories", "hk", "palmkernel", "gold", "ship", "soybean", "money-fx", "wheat", "fuel", "escudo", "ringgit", "ipi", "retail", "earn", "livestock", "cotton-meal", "yen", "l-cattle", "gnp", "coconut", "tea", "linseed", "bop", "peseta", "castor-oil", "cotton", "zinc", "tung-oil", "nickel", "cornglutenfeed", "reserves", "sun-oil"],
    "reuters-rcv1": ["M131", "GCRIM", "G153", "GDIP", "C312", "GCAT", "E31", "GSPO", "C313", "G155", "E312", "M142", "E61", "GFAS", "C17", "C16", "GVOTE", "G158", "E132", "GVIO", "C32", "C41", "M13", "E411", "C171", "G159", "E511", "E512", "C12", "GWELF", "E21", "G154", "E14", "GENV", "E12", "GDIS", "C331", "G156", "CCAT", "C182", "G152", "GSCI", "C411", "C152", "M132", "C33", "E142", "C21", "M12", "MCAT", "GWEA", "C311", "C151", "GTOUR", "C174", "C11", "GMIL", "E131", "C31", "E13", "C172", "GHEA", "M11", "C42", "C1511", "E311", "C24", "GPOL", "GJOB", "E51", "C13", "GOBIT", "G15", "C14", "G151", "C18", "GDEF", "E11", "E211", "C34", "GODD", "E143", "E41", "E212", "C181", "C173", "GREL", "GPRO", "C183", "GENT", "G157", "E71", "M14", "C23", "C22", "ECAT", "E141", "C15", "E121", "M143", "M141", "E313", "E513"],
    "athome1": ["athome102", "athome104", "athome105", "athome103", "athome107", "athome101", "athome109", "athome106", "athome108", "athome100"],
    "athome4": ["433", "413", "407", "410", "432", "416", "422", "414", "405", "434", "423", "403", "412", "409", "401", "426", "408", "402", "429", "418", "428", "427", "419", "430", "411", "415", "421", "417", "424", "404", "431", "425", "406", "420"]
}

down_sample_topics={
    "down-reuters": ["GJOB", "E12", "GDIS", "E11", "GVOTE", "C42", "E211", "E51", "G154", "M143", "C172", "M142", "GREL", "C14", "E14", "GODD", "GPRO", "C32", "G155", "E411", "E71", "C183", "E31", "C23", "GSCI", "E131", "E311", "GOBIT", "G157", "C331"],
    "down-tr4": ["415", "417", "419", "407", "416", "412", "410", "403", "430", "414", "429", "420", "425", "402", "413", "404", "424", "428", "423", "427", "401", "409", "418", "431", "432", "406", "405", "426", "408", "433"]
}

active_learning_mode_for_experiments = "ranked" # or "random"
# experiments = [
    # ("lemma_alpha", 0.95, 0.80, learning_rate_run, "vigilance .95 with slow recode beta 0.80 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_beta", 0.95, 0.85, learning_rate_run, "vigilance .95 with slow recode beta 0.85 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_gamma", 0.95, 0.90, learning_rate_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("lemma_delta", 0.95, 0.95, learning_rate_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
    # ("debug_run", 0.95, 1.0, learning_rate_run, "vigilance .95 with fast learn beta 1.0 Reuters, debug run", active_learning_mode_for_experiments)
# ]

experiments = [
    ("small_reuters_test_alpha", 0.95, 0.80, reuters_small_test, "vigilance .95 with slow recode beta 0.80 Reuters and 20 Newsgroups", active_learning_mode_for_experiments, False, 1),
#     ("rerun_beta", 0.60, 1.0, re_run, "vigilance .95 with slow recode beta 0.85 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_gamma", 0.90, 0.75, re_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_delta", 0.90, 1.0, re_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_epsilon", 0.95, 0.75, re_run, "vigilance .95 with slow recode beta 0.90 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
#     ("rerun_zeta", 0.95, 1.0, re_run, "vigilance .95 with slow recode beta 0.95 Reuters and 20 Newsgroups", active_learning_mode_for_experiments),
]

rho = 0.95
beta = 1.0
# experiments = []

prefixes = []
newsgroups_prefix = "bravo_golf"
reuters_small_prefix = "bravo_hotel"
chunk_size = 3

def build_exp_chunks(corpus, corpus_and_topics, param_prefix):
    chunk_size = 1
    for chunk_index, i in enumerate(range(0, len(corpus_and_topics[corpus]), chunk_size)):
        topics = {corpus: corpus_and_topics[corpus][i:i+chunk_size]}
        prefixes.append(f'"{param_prefix}_{chunk_index}"')
        experiments.append((f"{param_prefix}_{chunk_index}", rho, beta, topics, f"vigilance {rho} with beta {beta} {corpus} topics - {active_learning_mode_for_experiments} active learning mode", active_learning_mode_for_experiments, True))

# for chunk_index, i in enumerate(range(0, len(newsgroups), chunk_size)):
#     topics = {"20newsgroups": newsgroups[i:i+chunk_size]}
#     prefixes.append(f'"{newsgroups_prefix}_{chunk_index}"')
#     experiments.append((f"{newsgroups_prefix}_{chunk_index}", rho, beta, topics, f"vigilance {rho} with beta {beta} 20Newsgroups topics - {active_learning_mode_for_experiments} active learning mode", active_learning_mode_for_experiments))

# for chunk_index, i in enumerate(range(0, len(small_reuters_topics), chunk_size)):
#     topics = {"reuters21578": small_reuters_topics[i:i+chunk_size]}
#     prefixes.append(f'"{reuters_small_prefix}_{chunk_index}"')
#     experiments.append((f"{reuters_small_prefix}_{chunk_index}", rho, beta, topics, f"vigilance {rho} with beta {beta} Reuters-21578 topics - {active_learning_mode_for_experiments} active learning mode", active_learning_mode_for_experiments))

# reuters_down_sample_prefix = "bravo_india"
# athome_one_prefix = "kilo"
# athome_down_sample_prefix = "charlie_juliett"
# build_exp_chunks("down-reuters", down_sample_topics, reuters_down_sample_prefix)
# build_exp_chunks("down-tr4", down_sample_topics, athome_down_sample_prefix)
# build_exp_chunks("athome1", full_topics, athome_one_prefix)
# build_exp_chunks("")

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
# ranked: # ["bravo_echo_0", "bravo_echo_1", "bravo_echo_2", "bravo_echo_3", "bravo_echo_4", "bravo_echo_5", "bravo_echo_6", "bravo_foxtrot_0", "bravo_foxtrot_1", "bravo_foxtrot_2", "bravo_foxtrot_3", "bravo_foxtrot_4", "bravo_foxtrot_5", "bravo_foxtrot_6", "bravo_foxtrot_7", "bravo_foxtrot_8", "bravo_foxtrot_9", "bravo_foxtrot_10", "bravo_foxtrot_11", "bravo_foxtrot_12", "bravo_foxtrot_13", "bravo_foxtrot_14", "bravo_foxtrot_15", "bravo_foxtrot_16", "bravo_foxtrot_17", "bravo_foxtrot_18", "bravo_foxtrot_19", "bravo_foxtrot_20", "bravo_foxtrot_21", "bravo_foxtrot_22", "bravo_foxtrot_23", "bravo_foxtrot_24", "bravo_foxtrot_25", "bravo_foxtrot_26", "bravo_foxtrot_27", "bravo_foxtrot_28", "bravo_foxtrot_29", "bravo_foxtrot_30", "bravo_foxtrot_31", "bravo_foxtrot_32", "bravo_foxtrot_33", "bravo_foxtrot_34", "bravo_foxtrot_35", "bravo_foxtrot_36", "bravo_foxtrot_37", "bravo_foxtrot_38", "bravo_foxtrot_39", "bravo_foxtrot_40", "bravo_foxtrot_41", "bravo_foxtrot_42", "bravo_foxtrot_43", "bravo_foxtrot_44"]
# random: # ["bravo_golf_0", "bravo_golf_1", "bravo_golf_2", "bravo_golf_3", "bravo_golf_4", "bravo_golf_5", "bravo_golf_6", "bravo_hotel_0", "bravo_hotel_1", "bravo_hotel_2", "bravo_hotel_3", "bravo_hotel_4", "bravo_hotel_5", "bravo_hotel_6", "bravo_hotel_7", "bravo_hotel_8", "bravo_hotel_9", "bravo_hotel_10", "bravo_hotel_11", "bravo_hotel_12", "bravo_hotel_13", "bravo_hotel_14", "bravo_hotel_15", "bravo_hotel_16", "bravo_hotel_17", "bravo_hotel_18", "bravo_hotel_19", "bravo_hotel_20", "bravo_hotel_21", "bravo_hotel_22", "bravo_hotel_23", "bravo_hotel_24", "bravo_hotel_25", "bravo_hotel_26", "bravo_hotel_27", "bravo_hotel_28", "bravo_hotel_29", "bravo_hotel_30", "bravo_hotel_31", "bravo_hotel_32", "bravo_hotel_33", "bravo_hotel_34", "bravo_hotel_35", "bravo_hotel_36", "bravo_hotel_37", "bravo_hotel_38", "bravo_hotel_39", "bravo_hotel_40", "bravo_hotel_41", "bravo_hotel_42", "bravo_hotel_43", "bravo_hotel_44"]

# combined: # ["bravo_echo_0", "bravo_echo_1", "bravo_echo_2", "bravo_echo_3", "bravo_echo_4", "bravo_echo_5", "bravo_echo_6", "bravo_foxtrot_0", "bravo_foxtrot_1", "bravo_foxtrot_2", "bravo_foxtrot_3", "bravo_foxtrot_4", "bravo_foxtrot_5", "bravo_foxtrot_6", "bravo_foxtrot_7", "bravo_foxtrot_8", "bravo_foxtrot_9", "bravo_foxtrot_10", "bravo_foxtrot_11", "bravo_foxtrot_12", "bravo_foxtrot_13", "bravo_foxtrot_14", "bravo_foxtrot_15", "bravo_foxtrot_16", "bravo_foxtrot_17", "bravo_foxtrot_18", "bravo_foxtrot_19", "bravo_foxtrot_20", "bravo_foxtrot_21", "bravo_foxtrot_22", "bravo_foxtrot_23", "bravo_foxtrot_24", "bravo_foxtrot_25", "bravo_foxtrot_26", "bravo_foxtrot_27", "bravo_foxtrot_28", "bravo_foxtrot_29", "bravo_foxtrot_30", "bravo_foxtrot_31", "bravo_foxtrot_32", "bravo_foxtrot_33", "bravo_foxtrot_34", "bravo_foxtrot_35", "bravo_foxtrot_36", "bravo_foxtrot_37", "bravo_foxtrot_38", "bravo_foxtrot_39", "bravo_foxtrot_40", "bravo_foxtrot_41", "bravo_foxtrot_42", "bravo_foxtrot_43", "bravo_foxtrot_44", "bravo_golf_0", "bravo_golf_1", "bravo_golf_2", "bravo_golf_3", "bravo_golf_4", "bravo_golf_5", "bravo_golf_6", "bravo_hotel_0", "bravo_hotel_1", "bravo_hotel_2", "bravo_hotel_3", "bravo_hotel_4", "bravo_hotel_5", "bravo_hotel_6", "bravo_hotel_7", "bravo_hotel_8", "bravo_hotel_9", "bravo_hotel_10", "bravo_hotel_11", "bravo_hotel_12", "bravo_hotel_13", "bravo_hotel_14", "bravo_hotel_15", "bravo_hotel_16", "bravo_hotel_17", "bravo_hotel_18", "bravo_hotel_19", "bravo_hotel_20", "bravo_hotel_21", "bravo_hotel_22", "bravo_hotel_23", "bravo_hotel_24", "bravo_hotel_25", "bravo_hotel_26", "bravo_hotel_27", "bravo_hotel_28", "bravo_hotel_29", "bravo_hotel_30", "bravo_hotel_31", "bravo_hotel_32", "bravo_hotel_33", "bravo_hotel_34", "bravo_hotel_35", "bravo_hotel_36", "bravo_hotel_37", "bravo_hotel_38", "bravo_hotel_39", "bravo_hotel_40", "bravo_hotel_41", "bravo_hotel_42", "bravo_hotel_43", "bravo_hotel_44"]

# ranked down-reuters: ["india_0", "india_1", "india_2", "india_3", "india_4", "india_5", "india_6", "india_7", "india_8", "india_9", "india_10", "india_11", "india_12", "india_13", "india_14", "india_15", "india_16", "india_17", "india_18", "india_19", "india_20", "india_21", "india_22", "india_23", "india_24", "india_25", "india_26", "india_27", "india_28", "india_29"]
# random down-reuters: ["bravo_india_0", "bravo_india_1", "bravo_india_2", "bravo_india_3", "bravo_india_4", "bravo_india_5", "bravo_india_6", "bravo_india_7", "bravo_india_8", "bravo_india_9", "bravo_india_10", "bravo_india_11", "bravo_india_12", "bravo_india_13", "bravo_india_14", "bravo_india_15", "bravo_india_16", "bravo_india_17", "bravo_india_18", "bravo_india_19", "bravo_india_20", "bravo_india_21", "bravo_india_22", "bravo_india_23", "bravo_india_24", "bravo_india_25", "bravo_india_26", "bravo_india_27", "bravo_india_28", "bravo_india_29"]

# ranked down-tr4: ["juliett_0", "juliett_1", "juliett_2", "juliett_3", "juliett_4", "juliett_5", "juliett_6", "juliett_7", "juliett_8", "juliett_9", "juliett_10", "juliett_11", "juliett_12", "juliett_13", "juliett_14", "juliett_15", "juliett_16", "juliett_17", "juliett_18", "juliett_19", "juliett_20", "juliett_21", "juliett_22", "juliett_23", "juliett_24", "juliett_25", "juliett_26", "juliett_27", "juliett_28", "juliett_29"]
# random down-tr4: ["bravo_juliett_0", "bravo_juliett_1", "bravo_juliett_2", "bravo_juliett_3", "bravo_juliett_4", "bravo_juliett_5", "bravo_juliett_6", "bravo_juliett_7", "bravo_juliett_8", "bravo_juliett_9", "bravo_juliett_10", "bravo_juliett_11", "bravo_juliett_12", "bravo_juliett_13", "bravo_juliett_14", "bravo_juliett_15", "bravo_juliett_16", "bravo_juliett_17", "bravo_juliett_18", "bravo_juliett_19", "bravo_juliett_20", "bravo_juliett_21", "bravo_juliett_22", "bravo_juliett_23", "bravo_juliett_24", "bravo_juliett_25", "bravo_juliett_26", "bravo_juliett_27", "bravo_juliett_28", "bravo_juliett_29"]