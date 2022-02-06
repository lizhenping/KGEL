import torch
from model.utils import convert_examples_to_features, read_squad_examples
from transformers import GPT2Tokenizer
from model.GPT2LMHeadModel import GPT2LMHeadModel,GPT2Config
from beam_search.sample import sample_sequence
from data.feature import InputFeaturesQG, Example
##模型文件位置配置
output_config_file = "/home/DGNet/config.json"
# output_model_file = "/home/DGNet/pre_model/pytorch_model.bin"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

config = GPT2Config.from_json_file(output_config_file)
model = GPT2LMHeadModel(config)
tokenizer.add_special_tokens({'sep_token': '[SEP]'})
model.resize_token_embeddings(len(tokenizer))
# state_dict = torch.load(output_model_file)
# model.load_state_dict(state_dict)
checkpoint = torch.load("model_44")
model.load_state_dict(checkpoint['model_state_dict'])

# add the EOS token as PAD token to avoid warnings


train_features = convert_examples_to_features(is_training=True)
file_write_obj = open('/home/DGNet/output_predict.txt', 'w+',encoding="utf-8")
for step, batch in enumerate(train_features):
    # generate text until the output length (which includes the context length) reaches 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        generated = 0
        out = sample_sequence(
            model=model, length=50,
            context=batch,
            start_token=None,
            batch_size=1,
            temperature=1, top_k=3, device=device
        )

        out = out.tokens
        # print(out)
        questions = out
        text = tokenizer.decode(questions)
        #test2 = out[404:]
        test2 = out[404:]

        question_gold = batch.context_len_idxs.tolist()
        question_gold = question_gold[0][404:]
        #question_gold = question_gold[0]

        question_gold_text = tokenizer.decode(question_gold)
        question_pred_text = tokenizer.decode(test2)

        print("=" * 50 + " BEGIN " + str(generated) + " " + "=" * 50)
        end = "<"
        #question = "["

        # print(question_pred_text[:question_pred_text.index(end)])
        try:
            print(question_pred_text[:question_pred_text.index(end)])
            #print(question_pred_text)
        except Exception as e:
            print(question_pred_text)
        print(question_gold_text[:question_gold_text.index(end)])
        #print(question_gold_text)
        try:
            file_write_obj.write(question_pred_text[:question_pred_text.index(end)] + "\n")
        except Exception as e:
            file_write_obj.write(question_pred_text + "\n")
        try:
            file_write_obj.write(question_gold_text[:question_gold_text.index(end)] + "\n")
        except Exception as e:
            file_write_obj.write(question_gold_text + "\n")


        print("=" * 50 + " END " + str(generated) + " " + "=" * 50)


