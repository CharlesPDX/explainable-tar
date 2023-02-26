__author__ = "Leif Azzopardi"


import os
import sys
import re
from trec_eval.measures.tar_rulers import TarRuler, TarAggRuler
from trec_eval.seeker.trec_qrel_handler import TrecQrelHandler

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(results_file, qrel_file):

    qrh = TrecQrelHandler(qrel_file)
    print(qrh.get_topic_list()) # show what qrel topics have been read in
    #print( len(qrh.get_topic_list())) # show how many


    def get_value_and_check(qrh,seen_dict, topic_id, doc_id):
        # checks to make sure the document is in the qrels and was retrieved by the pubmed query
        v = 0
        if doc_id in seen_dict:
            logger.info("{} Duplicate {}".format(topic_id, doc_id))
            v = None
        else:
            seen_dict[d] = 1
            v = qrh.get_value(topic_id, doc_id)
            if v >= 3:
                v = None
        return v

    skip_topic = False
    curr_topic_id = ""
    seen_dict = {}

    tml = []
    tar_ruler = None
    scores = {}
    with open(results_file,"r") as rf:
        while rf:
            line = rf.readline()
            if not line:
                break
            #(topic_id, action,doc_id, rank, score, team) = re.split(" |\t",line)
            (topic_id, action,doc_id, rank, score, team) = line.split()

            if (topic_id == curr_topic_id):
                if (skip_topic is False):
                    # accumulate
                    #v = qrh.get_value(curr_topic_id, doc_id.strip())
                    d = doc_id.strip()
                    v = get_value_and_check(qrh, seen_dict, curr_topic_id, d)

                    if v is not None:
                        tar_ruler.update(v,v,action)
                else:
                    continue
            else:
                if curr_topic_id != "":
                    if skip_topic is False:

                        tar_ruler.finalize()
                        tml.append(tar_ruler)
                        tar_ruler.print_scores()
                    else:
                        skip_topic = False

                # new topic
                curr_topic_id = topic_id
                dl = qrh.get_doc_list(topic_id)
                num_docs = len(dl)
                num_rels = 0
                num_rels_in_set = 0
                num_docs_in_set = num_docs
                for d in dl:
                    val = qrh.get_value(topic_id, d)
                    if val > 0:
                        num_rels = num_rels + 1
                    if val == 1 or val == 2:
                        num_rels_in_set = num_rels_in_set + 1

                    if val == -1 or val > 2:
                        num_docs_in_set = num_docs_in_set - 1

                if (num_rels_in_set == 0):
                    logger.info(f"Skipping topic: {curr_topic_id} due to zero rels in set")
                    skip_topic = True
                    continue
                
                #print("D: {0} DS: {1} R: {2} RS: {3} ".format(num_docs,num_docs_in_set,num_rels, num_rels_in_set))
                tar_ruler = TarRuler(topic_id, num_docs_in_set, num_rels_in_set)

                # reset seen list
                seen_dict = {}
                d = doc_id.strip()


                v = get_value_and_check(qrh, seen_dict, topic_id, d)
                if v is not None:
                    tar_ruler.update(v,v,action)
        try:
            if skip_topic is False:
                tar_ruler.finalize()
                tml.append(tar_ruler)
                tar_ruler.print_scores()
                scores |= tar_ruler.yield_scores()

            agg_tar = TarAggRuler()
            for tar in tml:
                agg_tar.update(tar)
            agg_tar.finalize()
            agg_tar.print_scores()
            scores |= agg_tar.yield_scores()
            return scores
        except Exception as e:
            logger.error(e)
            pass

        



def usage(args):
    print("Usage: {0} <qrel_file> <results_file>".format(args[0]))


if __name__ == "__main__":
    # main('/home/ccourc/auto-stop-tar/ret/clef2017/tar_run/autotar-sp1.0-srNone-ct[]-csflr-md2-c1.0/CD008803/0/CD008803.run','/home/ccourc/auto-stop-tar/tar-master/2017-TAR/testing/qrels/qrel_content_test.txt')
    main(
        # '/home/ccourc/auto-stop-tar/ret/20newsgroups/tar_run/autotar-sp1.0-srNone-ct[]-csflr-md2-c1.0/alt.atheism/0/alt.atheism.run',
    "/home/ccourc/auto-stop-tar/ret/20newsgroups/tar_run/fam-sp1.0-srNone-/alt.atheism.short/0/alt.atheism.short.run",
    '/home/ccourc/auto-stop-tar/data/20newsgroups/qrels/alt.atheism.short')
    # filename = None
    # format = "TOP"

    # print(sys.argv)
    # if len(sys.argv) >= 2:
    #     qrels = sys.argv[1]

    # if len(sys.argv)==3:
    #     results = sys.argv[2]
    # else:
    #     usage(sys.argv)
    #     exit(1)
    # print(os.path.exists( results ))
    # if os.path.exists( results ) and os.path.exists(qrels):
    #     # main(results,qrels)
    # else:
    #     usage(sys.argv)
