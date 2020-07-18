'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import tweepy

import requests

main_url = " https://newsapi.org/v1/articles?source=bbc-news&sortBy=top&apiKey=4dbc17e007ab436fb66416009dfb59a8"
# personal details
consumer_key ="HSpcnhLn1vAZkipUUxRh6uWPh"
consumer_secret ="HFgIHu0gG2eS4SDpsMitN2P0sVHUTXToUxDgfrtg61pj1tol2a"
access_token ="1178619424905076741-aNhvEhvO4hE3Zjn39pBXUdF11WR3sR"
access_token_secret ="LyZ3JE8ZfoliL7KUda8ISN4txjy8Mn1xzUWcEOEjTpqze"

# authentication of consumer key and secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# authentication of access token and secret
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    #parser.add_argument("--text", type = file, required=True)
    parser.add_argument('filename')

    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    open_bbc_page = requests.get(main_url).json()

    article = open_bbc_page["articles"]

    results = []

    for ar in article:
        results.append(ar["title"])

    print(results[1])
    text1 = results[1]
    with open(args.filename) as file:
        #text1 = file.read()
        print(text1)

        if args.quiet is False:
            print(args)

        if args.batch_size == -1:
            args.batch_size = 1
        assert args.nsamples % args.batch_size == 0

        seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        enc = get_encoder()
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        model = load_weight(model, state_dict)
        model.to(device)
        model.eval()

        if args.length == -1:
            args.length = config.n_ctx // 2
        elif args.length > config.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

        print(text1)
        context_tokens = enc.encode(text1)

        generated = 0
        for _ in range(args.nsamples // args.batch_size):
            out = sample_sequence(
                model=model, length=args.length,
                context=context_tokens  if not  args.unconditional else None,
                start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device
            )
            out = out[:, len(context_tokens):].tolist()
            for i in range(args.batch_size):
                generated += 1
                text = enc.decode(out[i])
                if args.quiet is False:
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                text = text1 + text
                api.update_status(status = text)

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
