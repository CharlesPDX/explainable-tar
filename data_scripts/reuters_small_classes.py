from email.policy import default
import os
from pathlib import Path
from collections import defaultdict


def get_qrel_files(full_path):
    for root, _, file in os.walk(full_path):
        for file in file:
            yield os.path.join(root,file)


def get_relevance_counts(corpus_location):
    qrels_path = os.path.join(corpus_location, "qrels")
    if not Path(qrels_path).exists():
        print("qrels not found")
        return
    total_count_by_topic = defaultdict(int)
    relevant_count_by_topic = defaultdict(int)

    for qrel_file in get_qrel_files(qrels_path):
        with open(qrel_file, "r") as qrel:
            for line in qrel:
                if len(line.split()) != 4:
                    continue
                topic_id, dummy, doc_id, rel = line.split()
                total_count_by_topic[topic_id] += 1
                if int(rel):
                    relevant_count_by_topic[topic_id] += 1
    
    return total_count_by_topic, relevant_count_by_topic


if __name__ == "__main__":
    corpus_location = "./data/reuters21578"
    total_count_by_topic, relevant_count_by_topic = get_relevance_counts(corpus_location)
    # print(total_count_by_topic)
    # print(relevant_count_by_topic)
    for topic, total_count in total_count_by_topic.items():
        if topic in relevant_count_by_topic:
            print(f"{topic} - {relevant_count_by_topic[topic]/total_count:.2%}")
        else:
            print(f"{topic} - 0.00%")
