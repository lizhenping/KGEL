class HotpotExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 answer_text=None,):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = answer_text

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 context_len_idxs,
                 context_idxs,
                 context_mask,
                 y1,
                 y2,
                 answer_len,
                 unique_id,
                 context,
                 answer,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 query_ids,
                 entity_mapping,
                 entity_length,
                 entity_mask,
                 adj,
                 answer_mapping
                 ):
        self.context_len_idxs = context_len_idxs
        self.context_idxs = context_idxs
        self.context_mask = context_mask
        self.y1 = y1,
        self.y2= y2,
        self.answer_len = answer_len
        self.context = context
        self.answer = answer
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.query_ids = query_ids
        self.entity_mapping = entity_mapping
        self.entity_length = entity_length
        self.entity_mask = entity_mask
        self.adj = adj
        self.answer_mapping = answer_mapping




