
import sys
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig
from calibrate.evaluate import evaluate
import models

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main(args):
    # Initialization
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    args.device = torch.device("cuda")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if "llama" in args.model_name_or_path.split("/")[-1].lower():
        ClusCompModel = models.ClusterLayerLlamaForCausalLM
    elif "mistral" in args.model_name_or_path.split("/")[-1].lower():
        ClusCompModel = models.ClusterLayerMistralForCausalLM
    elif "opt" in args.model_name_or_path.split("/")[-1].lower():
        ClusCompModel = models.ClusterLayerOPTForCausalLM

    model = ClusCompModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        device_map='cpu', 
        #torch_dtype=torch.bfloat16
        torch_dtype=config.torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    assert args.seqlen <= config.max_position_embeddings, "The sequence length of calibration samples exceed the model's"
    model.eval()
    logging.info(model)
    evaluate(model, tokenizer, args, logging)

def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    # Calibration data
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length of calibration sample")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache dir of dataset, leading to faster debug")
    parser.add_argument("--eval_ppl", default=True, action="store_false")
    parser.add_argument("--limit", type=int, default=-1, help="Number of samples in evaluation for debug purpose.")
    args = parser.parse_args()
    return args

def cli_main():
    args = arg_parse()
    logging.info(sys.argv)
    logging.info(args)
    main(args)

if __name__ == "__main__":
    cli_main()