import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model.utils import convert_examples_to_features, read_squad_examples
from tqdm import tqdm
from torch.nn import DataParallel
from transformers import GPT2Tokenizer
from numpy.random import shuffle
from transformers import GPT2Model, GPT2Config
from model.GPT2LMHeadModel import GPT2LMHeadModel
from data.feature import InputFeaturesQG, Example
import logging
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--train_file', default='data/yewu.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--epochs', default=45, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=24, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-3, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=100, type=int, required=False,
                        help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--train_batch_size', default=10, type=int, required=False, help='训练batch size')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')

    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)

    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='/home/DGNet/data', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    #model_config = GPT2Config()
    #print('config:\n' + model_config.to_json_string())

    #n_ctx = model_config.n_ctx
    full_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    num_add_tokens = full_tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)



    epochs = args.epochs

    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    #stride = args.stride
    gradient_accumulation = args.gradient_accumulation


    max_grad_norm = args.max_grad_norm

    min_length = args.min_length
    output_dir = args.output_dir
    #tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    epoch_start = 0
    if not args.pretrained_model:
        #model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
        output_config_file = "/home/DGNet/config.json"
        # # output_model_file = "/home/DGNet/pre_model/pytorch_model.bin"
        config = GPT2Config.from_json_file(output_config_file)
        model = GPT2LMHeadModel(config)
        #
        # state_dict = torch.load(output_model_file)
        # embedding_layer = model.resize_token_embeddings(len(full_tokenizer))
        # model.load_state_dict(state_dict)
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        embedding_layer = model.resize_token_embeddings(len(full_tokenizer))





        #model = GPT2LMHeadModel.from_pretrained('gpt2')

        model_config = GPT2Config()
        #model_config.vocab_size = len(full_tokenizer)
        #assert model.transformer.wte.weight.shape[0] == len(full_tokenizer)
        print('config:\n' + model_config.to_json_string())
    else:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(full_tokenizer))

    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    #train_examples = read_squad_examples(input_file=args.train_file, is_training=True)
    train_features = convert_examples_to_features(is_training=True)
    total_steps = int(
        len(train_features) ) * args.epochs




    logger.info("***** Running training *****")
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Num steps = %d", total_steps)


    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps,
                                                  num_training_steps =total_steps)
    checkpoint = torch.load("model_36")
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch_start = checkpoint['epoch']
    print(f'---------[INFO] Restarting Training from Epoch {epoch_start} -----------\n')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    device = torch.device("cuda")
    ## train model
    for epoch in range(epoch_start,epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))

        piece_num = 0
        shuffle(train_features)
        for step, batch in enumerate(train_features):

            #outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            outputs = model(input_ids = batch,labels="fine-tuning")
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:

                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (overall_step + 1) % log_step == 0:
                #tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    piece_num,
                    epoch ,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))


                running_loss = 0
            overall_step += 1
            piece_num += 1


        ## temp save the model
        #if not epoch % 1 and epoch != 0:
        if epoch % 1 == 0:
            print('training %d finished' % epoch)
            if not os.path.exists(output_dir + 'model_%d' % epoch):
                os.mkdir(output_dir + 'model_%d' % epoch)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },  'model_%d' % epoch)


    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    model_to_save.config.to_json_file(output_dir + 'final_config.json')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
