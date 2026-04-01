import mlx.core as mx
import sentencepiece as spm
from train_gpt_mlx import GPT, Hyperparameters


def load_model():
    args = Hyperparameters()
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    weights = dict(mx.load("logs/mlx_smoke_mlx_model.npz"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, args


def generate(model, sp, prompt="The", max_tokens=200, temperature=0.8):
    token_ids = sp.encode(prompt)
    input_ids = mx.array([token_ids])

    for _ in range(max_tokens):
        hidden = model(input_ids)
        last_hidden = hidden[:, -1, :]
        logits = last_hidden @ model.tok_emb.weight.T
        logits = model.softcap(logits)

        if temperature > 0:
            probs = mx.softmax(logits / temperature, axis=-1)
            next_token = mx.random.categorical(mx.log(probs))
        else:
            next_token = mx.argmax(logits, axis=-1)

        next_token = next_token.reshape(1, 1)
        input_ids = mx.concatenate([input_ids, next_token], axis=1)
        mx.eval(input_ids)

        if int(next_token.item()) == sp.eos_id():
            break

    return sp.decode(input_ids[0].tolist())


def main():
    model, args = load_model()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    prompts = [
        "The quick brown",
        "In the year 2025",
        "Scientists discovered that",
        "The president of",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt!r}")
        print(f"Output: {generate(model, sp, prompt=prompt, max_tokens=100)}")
        print()


if __name__ == "__main__":
    main()
