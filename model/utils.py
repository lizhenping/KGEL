import logging
import json
import collections
import math
import sys
from model.HotpotExample import HotpotExample, InputFeatures
from data.data_helper import DataHelper


from data.config import set_config
import torch
from data.feature import InputFeaturesQG, Example


logger = logging.getLogger(__name__)


def read_squad_examples(input_file, is_training, version_2_with_negative=True):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        if entry["type"]=="comparison":
            continue
        else:
            qas_id = entry["_id"]
            context = entry["context"]
            question_text = entry["question"]
            answer_text = entry["answer"]
            supporting_facts = entry["supporting_facts"]
            content = []
            for title in supporting_facts:
                for con in context:
                    if con[0] == title[0]:
                        text = ' '.join(con[1])
                        content.append(text)
            doc_tokens = ' '.join(content)



        example = HotpotExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            answer_text=answer_text,
         )
        examples.append(example)
    logger.warning("finish the data process")
    return examples


def convert_examples_to_features(is_training=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 0

    features = []
    args = set_config()
    use_cuda = torch.cuda.is_available()
    dataloader = DataHelper(gz=True, config=args)
    if is_training == True:
        train_dataloader = dataloader.train_loader
    else:
        train_dataloader = dataloader.dev_loader
    # for (example_index, example) in enumerate(train_dataloader):
    #     context = example["context_idxs"]
    #     context2 = context.tolist()
    #     context2 = context2[0]
    #     #question = example.tolist()
    #     question = example["tgt_idxs"]
    #     question = question[0]
    #     question2 = question.tolist()
    #     answer = example["ans_idxs"]
    #     answer2 = answer.tolist()
    #     answer2 = answer2[0]
    #     print("test")

    # question_token = tokenizer.encode("[QUESTION]")
    # sep_token = tokenizer.encode("[SEP]")
    # answer_token = tokenizer.encode("[ANSWER]:")
    # end_token = tokenizer.encode("<|endoftext|>")
    # n_token = tokenizer.encode("\n")
    a = 0
    for (example_index, example) in enumerate(train_dataloader):
        # a = a +1
        # if a>50:
        #      break
        all_doc_tokens = example["context_len_idxs"].clone()
        #
        # context = example["context_idxs"].tolist()[0]
        # context2 = example["context_idxs"].tolist()[1]
        # context1 = len(context)
        # context2 = len(context2)
        # try:
        #     context = context[:context.index(0)]
        # except ValueError:
        #     print("context")
        #     pass

        # try:
        #     answer = answer[:answer.index(0)]
        # except ValueError:
        #     print("answer")
        #     pass


        #context2_text = tokenizer.decode(context)
        # question = example.tolist()


        # query_tokens = context + answer + sep_token + n_token + answer_token +n_token
        # query_tokens_text = tokenizer.decode(query_tokens)

        #question = "[QUESTION]"+example.doc_tokens + "[SEP]" + example.orig_answer_text + "[SEP]" + "\n" + "[ANSWER]:"
        #query_tokens = tokenizer.encode(question)

        #answer_text = tokenizer.encode(example.orig_answer_text)

        #     question = question[:question.index(0)]
        # except ValueError:
        #     print("question")
        #     pass
        question1 = example["tgt_idxs"].clone()
        entity_mapping = example["entity_mapping"].clone()
        entity_length = example["entity_lens"].clone()
        entity_mask = example["entity_mask"].clone()
        adj = example["entity_graphs"].clone()
        answer_mapping = example["ans_mask"].clone()
        context_len_idxs = example["context_len_idxs"].clone()
        context_idxs = example["context_idxs"].clone()
        context_mask = example["context_mask"].clone()
        y1 = example["y1"].clone()
        y2 = example["y2"].clone()

        answer_len = example["answer_len"]


        #batch = (entity_mapping,entity_length,entity_mask,adj,answer_mapping,context_len_idxs,answer_len)
        #all_doc_tokens = context + answer + sep_token + n_token + answer_token + question + end_token + n_token
        #all_doc_tokens_text = tokenizer.decode(all_doc_tokens)
        #answer = "[QUESTION]"+example.doc_tokens + "[SEP]" + example.orig_answer_text + "[SEP]" + "\n" + "[ANSWER]:" + example.question_text +"<|endoftext|>"+"\n"
        #all_doc_tokens = tokenizer.encode(answer)
        # while len(all_doc_tokens) < 450:
        #     all_doc_tokens.append(0)
        # if len(all_doc_tokens) > 450:
        #     print("+++++++++++++++++++")
        #     print(len(all_doc_tokens))
        #     all_doc_tokens = all_doc_tokens[:450]
        #     print(len(all_doc_tokens))
        #     print("+++++++++++++++++++")



        input_ids = all_doc_tokens
        query_ids = all_doc_tokens





        features.append(
            InputFeatures(
                context_len_idxs = context_len_idxs,
                context_idxs = context_idxs,
                context_mask= context_mask,
                y1=y1,
                y2=y2,
                answer_len = answer_len,
                context=context_len_idxs,
                answer=input_ids,
                unique_id=question1,
                example_index=example_index,
                doc_span_index=None,
                tokens=None,
                token_is_max_context=None,
                input_ids=input_ids,
                input_mask=None,
                segment_ids=None,
                query_ids=query_ids,
                entity_mapping=entity_mapping,
                entity_length = entity_length,
                entity_mask= entity_mask,
                adj = adj,
                answer_mapping = answer_mapping))

    return features




RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "query_tokens"])

