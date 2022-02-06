import torch
from torch import nn


def cal_position_id_2(input_ids, ans_start_ids, ans_end_ids, context_mask):
    # here we use the BOI scheme as described in  https://arxiv.org/pdf/1704.01792.pdf
    #assert input_ids.size(0) == ans_start_ids.size(0) == ans_end_ids.size(0)

    seq_length = input_ids.size(1)
    # position_ids = Variable(torch.arange(seq_length, dtype=torch.long, device=input_ids.device))
    position_ids = torch.zeros((input_ids.size(0), seq_length), dtype=torch.long, device=input_ids.device)

    for i in range(input_ids.size(0)):
        # print("the current start and end is {} and {}".format(ans_start_ids[i], ans_end_ids[i]))
        for j in range(seq_length):
            test = ans_start_ids[0][i]
            if j == ans_start_ids[0][i]:
                position_ids[i,j] = 1
            elif (j > ans_start_ids[0][i]) and (j <= ans_end_ids[0][i]):
                position_ids[i,j] = 2
            else:
                position_ids[i,j] = 0
    position_ids = position_ids * context_mask.type_as(position_ids)

    return position_ids
