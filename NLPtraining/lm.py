import os
import argparse
import logging
import torch
import random
import json
import numpy as np
from tqdm import tqdm
from logging import handlers
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from metric import get_generation_eval_scores
from openprompt import PromptForGeneration
from data_util import NLPRLProcessor, PromptDataLoader
from openprompt.utils.metrics import generation_metric
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from template import PrefixTuningTemplate
from torch.utils.tensorboard import SummaryWriter




class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射
 
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

        
def set_seed(args):
    """
    Set the random seed for reproducibility
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)       
def reshape(raw_list, num_return_sequences):  #from List(Str) to List(List(Str))  for multi generatation output
    reshaped_list = []
    i=0
    tmp_list = []
    for seq in raw_list:
        tmp_list.append(seq)
        i+=1
        if i == num_return_sequences:
            reshaped_list.append(tmp_list)
            i=0
            tmp_list = [] 
    return reshaped_list   
        
        
def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--plm_eval_mode", action="store_true")
    parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
    parser.add_argument("--prompt_type", type=str, default='prefix')
    parser.add_argument("--model_name_or_path", default='t5-base')
    parser.add_argument("--data_dir",default='./data/')
    parser.add_argument("--num_train_epochs", type=int, default= 3)
    parser.add_argument("--output_dir", type=str, default='./output/')
    parser.add_argument("--tensorboard_dir", type=str, default='tensorboard/')
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--check_step", type=int, default=500)
    parser.add_argument("--do_train", action="store_true", help="Whether to do training on the train set.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set at check steps during training.")
    parser.add_argument("--using_decoder_past_key_values", action="store_true", help="")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_debug", action="store_true", help="Whether to do debug.")
    parser.add_argument("--test_init_path", type=str, default=None, help="where saving the learned prompt parameters")
    parser.add_argument("--max_length", type=int, default=30, help="The maximal length of the generated sequences.")
    parser.add_argument("--min_length", type=int, default=1, help="The minimum length of the generated sequences.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of independently computed returned sequences for each element in the batch.")
    parser.add_argument("--beam_num", type=int, default=1, help="The number for beam search size.")   
    parser.add_argument("--num_prefix", type=int, default=5, help="The number of ") 
    parser.add_argument("--batch_size", type=int, default=8, help="The number of examples for one batch.") 
    parser.add_argument("--task", type=str, default="task0", help="The type of task")
    parser.add_argument("--level", type=str, default="high", help="The level of task")
    
    args = parser.parse_args()
    
    path = args.data_dir 
    
 #Load data
    dataset = {} 
    if args.do_train:
        dataset['train'] = NLPRLProcessor().get_examples(path,args.task, args.level)
        dataset['validation'] = NLPRLProcessor().get_examples(path, args.task, args.level)#[:100]
    if args.do_test:
        dataset['test'] = NLPRLProcessor().get_examples(path, args.task, args.level)#[:100]
    #dataset['validation'] = NLPRLProcessor().get_examples(os.path.join(path,'dev_50.tsv'),'dev')
   # dataset['test'] = NLPRLProcessor().get_examples(os.path.join(path,'test_eval_50.tsv'),'test')
 
    out_dir = os.path.join(
              args.output_dir, "lr{}-ep{}-sd{}".format(args.lr,args.num_train_epochs, args.seed)  
                )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log = Logger(f'{out_dir}/all.log',level='debug')
        
    set_seed(args)
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )    
        
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
    log.logger.info(f"The type of the backbone model:{args.model}   Load the model from:{args.model_name_or_path}   The random seed:{args.seed}")
    log.logger.info(f"The type of the prompt:{args.prompt_type}   Task:{args.task}   Level:{args.level}") 
    
 #set_template
    mytext ='{"placeholder":"text_a"} {"image"}{"tactile"} {"mask"}' 
    mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text=mytext,  num_token=args.num_prefix, using_decoder_past_key_values=True)
      
    if args.do_train:          
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
            batch_size=args.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
            truncate_method="head")

        validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
            batch_size=args.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
            truncate_method="head")

    if args.do_test:
        test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
            batch_size=args.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
            truncate_method="head")  

        
    prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)

    prompt_model=  prompt_model.to(device)        
        
    if args.do_train:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
        ]


        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

        tot_step  = len(train_dataloader)*args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)    

    generation_arguments = {
        "max_length": args.max_length,
        "max_new_tokens": None,
        "min_length": args.min_length,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,     #1.0 means no penalty
        "num_return_sequences":args.num_return_sequences,
        "num_beams": args.beam_num,
        "bad_words_ids": [[628], [198]]
    }
    
    def evaluate(prompt_model, dataloader, num_hyp, do_test):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
            
            reshaped_output_sentence = reshape(output_sentence, args.num_return_sequences)
            generated_sentence.extend(reshaped_output_sentence)

            groundtruth_sentence.extend(inputs['tgt_text'])
            
            #first_generated_sentence = [sent[0]for sent in generated_sentence]  #only use the first generated sentence to metrics         
        metrics, scores = get_generation_eval_scores(generated_sentence, groundtruth_sentence, num_hyp, do_test)    
       # bleu_score, rouge_score, bert_score = calc_score(first_generated_sentence, groundtruth_sentence)
        log.logger.info('\t'.join([f'{x}:{y}' for x,y in zip(metrics, scores)]))
#         log.logger.info(f'bleu_score:{bleu_score}, rouge_score:{rouge_score}, bert_score:{bert_score}')
#         score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
#         log.logger.info("test_score", score, flush=True)
        results = {x:float(y) for x,y in zip(metrics, scores)}
        return generated_sentence, groundtruth_sentence, results
        
    
    
    # training 
    if args.do_train:
        tb_writer = SummaryWriter(comment=f"{args.model_name_or_path.split('/')[-1]}-pf{args.num_prefix}-bs{args.batch_size}")
        global_step = 0 
        tot_loss = 0 
        log_loss = 0
        log.logger.info("************Training begin!******************")
        for epoch in range(args.num_train_epochs):
            prompt_model.train()
            for step, inputs in tqdm(enumerate(train_dataloader)):            
                global_step +=1
                inputs = inputs.to(device)
                loss = prompt_model(inputs)
                loss.backward()
                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if global_step %args.check_step ==0: 
                    log.logger.info("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/args.check_step, scheduler.get_last_lr()[0]))
                    log_loss = tot_loss
                    if args.do_eval:
                        prompt_model.eval()
                        _, _,results=evaluate(prompt_model, validation_dataloader, num_hyp=args.num_return_sequences, do_test=False)
                        prompt_model.train()
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss",(tot_loss-log_loss)/args.check_step,global_step)
                    log_loss = tot_loss
            #save results after each epoch
            prompt_model.eval()
            generated_sentence, groundtruch_sentence, results = evaluate(prompt_model, test_dataloader,num_hyp=args.num_return_sequences, do_test=True)
            prompt_model.train()
            with open(os.path.join(out_dir,f"{epoch}_results.txt"), 'w', encoding='utf-8') as f: 
                f.write(f"The type of the backbone model:{args.model}   Load the model from:{args.model_name_or_path}   The random seed:{args.seed}  Epoch:{epoch}\n")
                f.write(f"The type of the prompt:{args.prompt_type}   Task:{args.task}   Level:{args.level}\n")
                f.write('  '.join(f'{key}:{value}' for key, value in results.items()))
                f.write('\n')
                for y,z in zip(groundtruch_sentence, generated_sentence):
                    f.write(f'reference: {y}\n prediciton: {str(z)}\n')
                    f.write('\n')
        tb_writer.close()
        log.logger.info("Saving prompt parameters")
        prompt_model.eval()
        torch.save(prompt_model.template.state_dict(), os.path.join(out_dir, 'prompt_parameter.pt'))
    if args.do_test:
        log.logger.info("Loading prompt parameters")
        if not args.do_train:
            test_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
            test_model.template.load_state_dict(torch.load(os.path.join(args.test_init_path, 'prompt_parameter.pt')))
            #test_model.template.load_state_dict(torch.load(   "/home/syuanaf/Code/CometPrompt/output/few_shot_comet/250/prefix_plus_pp10_bm10_pf10/t5-xl-lm-adapt-bs32-lr5e-05/prompt_parameter(Jen3).pt"))
            test_model= test_model.to(device)
            test_model.eval()
            #save test model structure
#             with open(os.path.join(out_dir,'model_structure.txt'), 'w', encoding='utf-8') as f:
#                 print(test_model, file=f)  
                                  
            generated_sentence, groundtruch_sentence, results = evaluate(test_model, test_dataloader,num_hyp=args.num_return_sequences, do_test=True)
        #save output
            with open(os.path.join(args.output_dir,f'results.txt'), 'w', encoding='utf-8') as f: 
                f.write(f"The type of the backbone model:{args.model}   Load the model from:{args.model_name_or_path}   The random seed:{args.seed}\n")
                f.write(f"The type of the prompt:{args.prompt_type}   Task:{args.task}   Level:{args.level}")
                f.write('  '.join(f'{key}:{value}' for key, value in results.items()))
                f.write('\n')
                for y,z in zip(groundtruch_sentence, generated_sentence):
                    f.write(f' reference: {y}     |     prediciton: {str(z)}\n')                                  
                                  
                                  
                                  
        else:                         
            generated_sentence, groundtruch_sentence, results = evaluate(prompt_model, test_dataloader,num_hyp=args.num_return_sequences, do_test=True)
        #save output
            with open(os.path.join(out_dir,'test_results.txt'), 'w', encoding='utf-8') as f: 
                f.write(f"The type of the backbone model:{args.model}   Load the model from:{args.model_name_or_path}   The random seed:{args.seed}\n")
                f.write(f"The type of the prompt:{args.prompt_type}   Task:{args.task}   Level:{args.level}")
                f.write('  '.join(f'{key}:{value}' for key, value in results.items()))
                f.write('\n')
                for y,z in zip(groundtruch_sentence, generated_sentence):
                    f.write(f' reference: {y}     |     prediciton: {str(z)}\n')

            #save config
            config = {"plm_model":args.model, "plm_model_name_or_path":args.model_name_or_path, "text":mytext,  "num_prefix":args.num_prefix, "using_decoder_past_key_values":args.using_decoder_past_key_values}
            with open(os.path.join(out_dir,'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config,f)


if __name__ == "__main__":
    main()
    
    
        
 