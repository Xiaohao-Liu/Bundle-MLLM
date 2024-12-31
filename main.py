import os
import torch
import transformers
from transformers import EarlyStoppingCallback, Trainer
from tqdm import tqdm
import yaml
import argparse
import numpy as np

from model import BundleMLLM
from utils import Datasets, DataCollator, setup_seeds, preprocess_logits_for_metrics, compute_metrics

eps = 1e-9
MAP_LETTER = {
    i: idx for idx, i in
    enumerate([319, 350, 315, 360, 382, 383, 402, 379, 306, 435, 476, 365, 341, 405, 438, 349, 660, 390, 317, 323])
}
SHOW_A_OUTPUT_SAMPLE = False
TRAIN_ON_PROMPT_INPUTS = True


class Trainer2(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.llama_model.save_pretrained(output_dir)
        torch.save(self.model.projector.state_dict(), os.path.join(output_dir, "projector_model_fuse.bin"))
        torch.save(self.model.fusion.state_dict(), os.path.join(output_dir, "fusion.bin"))
        if self.model.conf["soft_prompt"]:
            torch.save(self.model.prompt_token.state_dict(), os.path.join(output_dir, "prompt_token.bin"))
        for i in [
            os.path.join(output_dir, "pytorch_model.bin"),
        ]:
            if os.path.exists(i):
                os.remove(i)


def train(conf, dataset):
    device = conf["device"]

    model = BundleMLLM(
        conf, dataset.graphs, dataset.features, dataset.item_info, is_training=True
    ).to(device)

    eval_step = 20

    model.print_trainable_params()
    output_dir = f"./checkpoint/{conf['dataset']}/{conf['mode']}/{conf['few_shot']}{conf['info']}"
    trainer = Trainer2(
        model=model,
        train_dataset=dataset.train_data,
        eval_dataset=dataset.test_data,
        args=transformers.TrainingArguments(
            warmup_steps=20,
            num_train_epochs=200,
            learning_rate=3e-4,
            per_device_train_batch_size=conf["batch_size_train"],
            per_device_eval_batch_size=conf["batch_size_train"],
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            run_name=f"SaMLLM_{conf['dataset']}_{conf['mode']}_{conf['few_shot']}{conf['info']}",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_hitrate",
            report_to=None,
            save_safetensors=False,

        ),
        compute_metrics=compute_metrics,
        data_collator=DataCollator(),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()


def test(conf, dataset):
    model = BundleMLLM(
        conf, dataset.graphs, dataset.features, dataset.item_info
    ).to(device)

    generation_config = transformers.GenerationConfig(
        temperature=0.01,
        top_p=0,
        top_k=10,
        num_beams=1,
        bos_token_id=model.tokenizer.bos_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        max_new_tokens=64,
        do_sample=True,
    )
    model.eval()

    log_path = conf["log_path"]
    with torch.no_grad():
        count_hit = 0
        count_valid = 0
        count_all = 0
        for b_id, seq_b_i_i, true_position, candidates in tqdm(dataset.test_loader):
            if conf["cot_stage"] == 1:
                outputs = model.evaluate(seq_b_i_i, candidates, generation_config)
            else:
                outputs, answers = model.evaluate_cot(seq_b_i_i, candidates, generation_config)

            seq_id = outputs.sequences
            output_seqs = model.tokenizer.batch_decode(seq_id, skip_special_tokens=True)

            for i in range(len(b_id)):
                pred_str = output_seqs[i].split(".")[0].strip()[0]
                true_indice = int(true_position[i]) + 1
                try:
                    pred_indice = ord(pred_str) - ord("A") + 1
                    if pred_indice > 0 and pred_indice <= 20:
                        count_valid += 1
                        count_hit += int(pred_indice == true_indice)
                    else:
                        pred_indice = -1
                except:
                    pred_indice = -1

                count_all += 1

                with open(log_path + ".outputs", "a") as f:
                    f.write(
                        f"{int(b_id[i])}, {true_indice}, {pred_indice}, {int(pred_indice == true_indice)}, {output_seqs[i].strip()}, \n"
                    )

        metrics = {
            "valid_ratio": count_valid / count_all,
            "hitrate": count_hit / count_valid
        }
        with open(log_path + ".results", "a") as f:
            str_ = ", ".join(f"{m}:{metrics[m]:.6f}" for m in metrics)
            f.write(str_)
            print(str_)

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0",
                        type=str, help="which gpu to use")
    parser.add_argument("-o", "--option", help="option: train or test", default="test", required=False)
    parser.add_argument("-d", "--dataset", default="pog", type=str, help="which dataset to use")
    parser.add_argument("-s", "--few_shot", default=1024, type=int, help="which dataset to use")
    parser.add_argument("-t", "--toy_eval", default=2048, type=int, help="which dataset to use")
    parser.add_argument("-n", "--num_token", default=5, type=int, help="maximum length of items in a squence")
    parser.add_argument("-i", "--info", default="", type=str, help="")
    parser.add_argument("-p", "--pretrain_model_path", default="", type=str, help="")
    parser.add_argument("-c", "--cot_stage", default=1, type=int)

    parser.add_argument("--num_cans", default=10, type=int,
                        help="the number of tokens (items in the bundle)")

    parser.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf", type=str,
                        help="lmsys/vicuna-7b-v1.5, meta-llama/Meta-Llama-3-8B ")

    parser.add_argument("--alpha", default=0, type=float,
                        help="")

    parser.add_argument("--test_log_path", default="", type=str,
                        help="")

    parser.add_argument("--components", default="", type=str,
                        help="layernorm, w_v")

    parser.add_argument("--soft_prompt", default=False, type=bool,
                        help="")

    parser.add_argument("--del_sp", default=False, type=bool,
                        help="")

    parser.add_argument("--sp_guided", default=False, type=bool,
                        help="")

    parser.add_argument("--sp_sg", default=False, type=bool,
                        help="stop_gradient")

    parser.add_argument("--train_lora", default="", type=str,
                        help="")

    parser.add_argument("--c_norm", default=False, type=bool,
                        help="")

    parser.add_argument("-m", "--mode", default="mm", type=str, help="mm, text, text+mm")

    parser.add_argument("--wandb", default="False", type=str, help="")

    paras = parser.parse_args().__dict__
    conf = yaml.safe_load(open("./config.yaml"))[paras["dataset"]]
    for p in paras:
        conf[p] = paras[p]

    print("load config file done!")

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    conf["from_pretrain"] = True if conf["pretrain_model_path"] != "" else False

    setup_seeds()

    if paras["option"] == "train":
        if conf["train_lora"] == "":
            conf["train_lora"] = True
        else:
            conf["train_lora"] = eval(conf["train_lora"])
        print(conf)
        os.environ["WANDB_DISABLED"] = "true" if conf["wandb"] == "False" else "false"
        train(conf, dataset)
    elif paras["option"] == "test":

        conf["train_lora"] = False
        conf["log_path"] = conf["test_log_path"] if conf[
                                                        "test_log_path"] != "" else f"./log/{conf['dataset']}_samllm_multi_{conf['mode']}_{conf['pretrain_model_path'].replace('/', '_')}{conf['info']}"

        metrics = {
            "valid_ratio": [],
            "hitrate": []
        }
        for i in range(1):
            dataset.test_loader.dataset.shift = i * conf["toy_eval"]  # i*256
            metric = test(conf, dataset)
            for m in metric:
                metrics[m].append(metric[m])

        with open(conf["log_path"] + ".results2", "a") as f:
            f.write(f"{conf['dataset'].upper()}/{conf['mode']}\n")
            str_ = ", ".join(f"{m}:{metrics[m]}" for m in metrics) + "\n"
            f.write(str_)
            print(str_)
            str_ = "; ".join(f"{m}: mean: {np.mean(metrics[m])} std: {np.std(metrics[m])}" for m in metrics) + "\n\n"
            f.write(str_)
            print(str_)
