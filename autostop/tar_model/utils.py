import os
from enum import Enum, auto


class MetricType(Enum):
    init = auto()
    step = auto()
    final = auto()


def calculate_ap(pid2label, ranked_pids, cutoff=0.5):
    num_rel = 0
    total_precision = 0.0
    for i, pid in enumerate(ranked_pids):
        label = pid2label[pid]
        if label >= cutoff:
            num_rel += 1
            total_precision += num_rel / (i + 1.0)

    return (total_precision / num_rel) if num_rel > 0 else 0.0


def calculate_losser(recall_cost, cost, N, R):
    return (1-recall_cost)**2 + (100/N)**2 * (cost/(R+100))**2


corpora_doctext_mapping = {
    '20newsgroups': '20newsgroups.collection.json',
    'reuters21578': 'reuters21578.collection.json',
    'athome1': 'athome1.collection.json',
    'athome4': 'athome4.collection.json',
    'legal': 'collection.json'
}

corpora_docid_mapping = {
    '20newsgroups': '20newsgroups.docids.txt',
    'reuters21578': 'reuters21578.docids.txt',
    'athome1': 'athome1.docids.txt',
    'athome4': 'athome4.docids.txt',
    'legal': 'docids.txt'
}

def make_file_params(collection_name: str, corpus_name: str, topic_id: str, topic_set: str) -> dict:
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    params = {'data_name': corpus_name,
    'topic_id': topic_id,
    'topic_set': topic_set,
    'query_file': os.path.join(PARENT_DIR, 'data', collection_name, 'topics', topic_id),
    'qrel_file': os.path.join(PARENT_DIR, 'data', collection_name, 'qrels', topic_id)}

    if corpus_name in corpora_doctext_mapping:
        params['doc_text_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'doctexts', corpora_doctext_mapping[corpus_name])
    else:
        params['doc_text_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'doctexts', topic_id)

    if corpus_name in corpora_docid_mapping:
        params['doc_id_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'docids', corpora_docid_mapping[corpus_name])
    else:
        params['doc_id_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'docids', topic_id)


    return params
