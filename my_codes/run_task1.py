
import os
import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import shutil
from tqdm import tqdm, trange
import json
import math
from transformers import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils_task1 import *
from bertmodel import BertForTask1
from utils_task1 import Task1DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
n_device = ["cuda: 0"]
assert len(n_device) == n_gpu

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, dev_dataset, model, tokenizer):
    # Overwrite the content of the output directory.
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        os.mkdir(args.output_dir)
    else:
        os.mkdir(args.output_dir)

    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
     # len=8000/batch, batch=gpu*batch_per_gpu
    t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer=optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Multi-gpu setting
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Instanteneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (parallel & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss, best_step = 0.0, 9999999999.0, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        pbar = tqdm(total=math.ceil(len(train_dataset) / args.train_batch_size), desc="Train Iteration")
        for step, batch in enumerate(train_dataset):
            model.train()
            pbar.update(1)
            inputs = {"labels":            batch[0][0],
                      "adjacent_matrix":   batch[1][0],
                      "concept_embedding": batch[2][0],
                      "entity_index":      batch[3][0],
                      "input_ids":         batch[4][0],
                      "attention_mask":    batch[5][0],
                      "token_type_ids":    batch[6][0]}

            outputs = model(**inputs)

            # evaluate的时候，两个句子预测的logit,都看他们取到1 即为错误的概率。谁的logit[1]更大， pred的label就是谁
            # 最后算pred label的时候多加一个函数。记录下来两句话各自logit[1]然后比较谁的logit[1]大
            loss = outputs["loss"]
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if ((step + 1) % args.gradient_accumulation_steps) == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate scheduler
                model.zero_grad()
                global_step += 1

                # 每50step，evaluate dev，保存最好的loss对应的step
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Eval on the dev set
                    results, _ = evaluate(args, dev_dataset, model, tokenizer)  # dev set evaluation
                    logger.info("***** Step %s: evaluate dev set results", str(global_step))

                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        logger.info("      %s = %s", key, value)
                    if results["eval_loss"] < best_dev_loss:  # 储存最低dev loss对应的step
                        best_dev_acc = results["eval_acc"]
                        best_dev_loss = results["eval_loss"]
                        best_step = global_step

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    outputs_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(outputs_dir):
                        os.makedirs(outputs_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(outputs_dir)
                    logger.info("Checkpoint {} saved at".format(global_step))
                    torch.save(args, os.path.join(outputs_dir, "training_args.bin"))
        pbar.close()
        train_dataset.reshuffle()
    logger.info("Min dev loss: %s, dev acc: %s, best global step: %s", str(best_dev_loss), str(best_dev_acc), str(best_step))
    tb_writer.close()
    return global_step, tr_loss / global_step, best_step


def evaluate(args, dataset, model, tokenizer, test=False):
    wrong_samples = []

    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Eval
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    pbar = tqdm(total=math.ceil(len(dataset) / args.eval_batch_size), desc="Test/Dev Iteration")
    for step, batch in enumerate(dataset):
        model.eval()
        pbar.update(1)
        with torch.no_grad():
            # inputs_ids [batch, 2, max_length]   labels [batch, 1\
            inputs = {"labels":            batch[0][0],
                      "adjacent_matrix":   batch[1][0],
                      "concept_embedding": batch[2][0],
                      "entity_index":      batch[3][0],
                      "input_ids":         batch[4][0],
                      "attention_mask":    batch[5][0],
                      "token_type_ids":    batch[6][0]}
            outputs = model(**inputs)
            loss, logits = outputs["loss"], outputs["logits"]  # logits [[batch, 2], [batch, 2]]

            if args.n_gpu > 1:
                # i.mean() 平均单个sent在所有gpu上的均值；Tensor.mean()计算两个sent的平均值
                loss = loss.mean()
            eval_loss += loss
        nb_eval_steps += 1

        if preds is None:
            pred = preds_decode(logits).int().numpy()  # 比较两句话的logit[1]的大小，谁更可能是错的，调用utils_task1里面的函数
            label = inputs["labels"].detach().cpu().numpy()

            preds = pred
            out_label_ids = label
        else:
            pred = preds_decode(logits).int().numpy()
            label = inputs["labels"].detach().cpu().numpy()

            preds = np.append(preds, pred, axis=0)
            out_label_ids = np.append(out_label_ids, label, axis=0)

        compare = pred == label
        # for i in range(len(compare)):
        #     if not compare[i]:
        #         wrong_samples.append(inputs["guids"].detach().cpu().numpy()[i])


    eval_loss = eval_loss / nb_eval_steps
    acc = simple_accuracy(preds, out_label_ids)

    result = {"eval_acc": acc, "eval_loss": eval_loss}
    results.update(result)

    return results, wrong_samples


def main():
    # [1] Parameters Setting
    parser = argparse.ArgumentParser()
    # File directory setting
    parser.add_argument("--data_dir", default="./dataset/task1/", type=str, help="Input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--output_dir", default="./model/fine_tuned/", type=str, help="Output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_name", default="_l_exp", type=str, help="Prefix of .tsv files. If default, then train.tsv")
    # parser.add_argument("--overwrite_output_dir", default=True, help="Overwrite the content of the output directory.")
    parser.add_argument("--wrong_file", default="./eval_result/solo_w_loss.csv", type=str, help="Save the wrong predicted samples.")

    # Checkpoint saving setting
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X update steps.")

    # Model parameters setting
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length", default=80, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--n_gpu", default=n_gpu, type=int, help="Num of gpus to use. Equal to the length of CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Train or evaluate setting
    # parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, help='Whether to run test on the test set')
    parser.add_argument("--eval_step", default=None, type=int, help="If don't run, only test the dev/test set on the specifix checkpoint model w.r.t eval_step.")

    args = parser.parse_args()

    # [2] Set up CUDA and GPU device
    device = torch.device("cuda")
    # device = torch.device("cpu")
    args.device = device
    logger.info("Training/evaluation parameters %s", args)

    # [3] Prepare task
    set_seed(args)

    # [4] Load pretrained Bert model and tokenizer https://zhuanlan.zhihu.com/p/50773178
    config = BertConfig.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForTask1.from_pretrained("bert-base-uncased", config=config)
    model.to(device)

    # [5] Training and save best-practice. Only when eval_step is None and need train.
    if args.eval_step is None:
        train_dataset = Task1DataLoader(sent_trp_jsonl=args.data_dir, batch_size=args.per_gpu_train_batch_size,
                                        n_devices=n_device, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                        data_name=args.data_name)
        dev_dataset = Task1DataLoader(sent_trp_jsonl=args.data_dir, batch_size=args.per_gpu_train_batch_size,
                                        n_devices=n_device, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                        data_name=args.data_name, dev=True)

        global_step, tr_loss, best_step = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" Training global_step = %s", global_step)
        logger.info(" Training average loss = %s", tr_loss)
        logger.info(" Best global step = %s", best_step)
        args.eval_step = best_step
    else:
        pass

    # [6] Evaluation on the test dataset with the best checkpoint from train or manually specified
    if args.do_test:
        # checkpoint = os.path.join(args.output_dir, "checkpoint-{}".format(str(args.eval_step)))
        # model_eval = BertForTask1.from_pretrained(checkpoint, config=config)
        # model_eval.to(args.device)
        # test_results, wrong_samples = evaluate(args, model_eval, tokenizer, test=True)
        # logger.info("Corresponding test loss: %s, test acc: %s, eval step: %s", str(test_results["eval_loss"]), str(test_results["eval_acc"]), str(args.eval_step))
        #
        # logger.info("Save %d wrong samples into file: %s", len(wrong_samples), str(args.wrong_file))

        checkpoint = os.path.join(args.output_dir, "checkpoint-{}".format(str(args.eval_step)))
        model_eval = BertForTask1.from_pretrained(checkpoint, config=config)
        model_eval.to(device)

        test_device = ["cuda:0"]
        args.test_batch_size = args.per_gpu_eval_batch_size
        test_dataset = Task1DataLoader(sent_trp_jsonl=args.data_dir, batch_size=args.per_gpu_train_batch_size,
                                        n_devices=n_device, tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                                        data_name=args.data_name, test=True, shuffle=False)

        test_results, wrong_samples = evaluate(args, dataset=test_dataset, model=model_eval, tokenizer=tokenizer, test=True)
        logger.info("Corresponding test loss: %s, test acc: %s, eval step: %s", str(test_results["eval_loss"]),
                    str(test_results["eval_acc"]), str(args.eval_step))

        logger.info("Save %d wrong samples into file: %s", len(wrong_samples), str(args.wrong_file))



if __name__ == '__main__':
    main()