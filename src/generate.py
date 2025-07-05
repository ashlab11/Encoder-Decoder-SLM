#Starts up a CLI to talk to the model
#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from components.base.basic_model import BasicEDModel

def main():
    p = argparse.ArgumentParser(description="Chat CLI for your ED model")
    p.add_argument("--model-path",    default="models/base/sft.pt", help="Path to your saved model checkpoint")
    p.add_argument("--tokenizer",     default="tokenizer/tokenizer.json", help="Path to your tokenizer JSON (e.g. tokenizer.json)")
    p.add_argument("--device",        default="cuda:0",   help="torch device, e.g. 'cpu' or 'cuda:0'")
    p.add_argument("--max-new-tokens",type=int, default=128, help="Max tokens per reply")
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--top-k",         type=int,   default=50)
    p.add_argument("--top-p",         type=float, default=0.9)
    args = p.parse_args()

    # 1) load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # 2) instantiate & load your model
    device = torch.device(args.device)
    model = BasicEDModel(
        vocab_size=tokenizer.get_vocab_size(),
        dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=3,
        enc_seq_len=1024,
        dec_seq_len=512,
        pad_token_id=tokenizer.token_to_id("<pad>")
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device).eval()
    
    #Print num params
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    print("\n🦙  Chat started; type your message and hit enter.  (Ctrl-D to quit)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            # prepend role tag if you use one
            prompt = f"<user> {user_input}"
            # run your generate() exactly as you defined it:
            reply = model.generate(
                input=prompt,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device
            )
            print(f"Bot: {reply}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye! 👋")

if __name__ == "__main__":
    main()
