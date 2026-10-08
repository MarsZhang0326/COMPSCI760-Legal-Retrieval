"""Exact archived E5 training/encoding helpers; use the separate worker entry scripts."""
from __future__ import annotations
from pathlib import Path
import collections, hashlib, json, math, os, random, re, sys, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def save_json(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def clean_query(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def query_input(q):
    return "query: " + clean_query(q)

def device(c):
    import torch
    if c["device"] != "auto":
        return torch.device(c["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_encoder(c, path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(c["threads"])
    model = AutoModel.from_pretrained(path,local_files_only=True).to(device(c))
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
    return model, tokenizer

def embeddings(model, tokenizer, texts, c):
    import torch
    import torch.nn.functional as F
    batch = tokenizer(texts,padding=True,truncation=True,max_length=c["max_length"],return_tensors="pt")
    batch = {k:v.to(next(model.parameters()).device) for k,v in batch.items()}
    h = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),p=2,dim=1)

def encode(model, tokenizer, texts, c):
    import torch
    import contextlib
    model.eval()
    out = []
    ctx = torch.autocast("cuda",dtype=torch.float16) if c["amp"] and next(model.parameters()).device.type == "cuda" else contextlib.nullcontext()
    with torch.no_grad(),ctx:
        for i in range(0,len(texts),c["encode_batch_size"]):
            out.append(embeddings(model,tokenizer,texts[i:i+c["encode_batch_size"]],c).float().cpu().numpy())
    return np.concatenate(out)

def contrastive_masks(batch, docs):
    """Known gold of ANY query is positive for that query, never a false negative.
    Group-only labels are masked (not promoted to positives). Duplicate positives
    from other queries are included in the multi-positive numerator.
    """
    positive = np.asarray([[d["citation"] in r["gold"] for d in docs] for r in batch],dtype=bool)
    allowed = np.asarray([[d["citation"] not in r["exclude_gold"] or d["citation"] in r["gold"]
                           for d in docs] for r in batch],dtype=bool)
    for i,r in enumerate(batch):
        ph = digest(clean_query(r["positive"]["text"]))
        for j,d in enumerate(docs):
            if not positive[i,j] and digest(clean_query(d["text"])) == ph:
                allowed[i,j] = False
    assert positive.any(axis=1).all()
    assert (allowed | ~positive).all()
    return positive,allowed

def multi_positive_loss(logits, positive, allowed):
    import torch
    positive = torch.as_tensor(positive,device=logits.device)
    allowed = torch.as_tensor(allowed,device=logits.device)
    return (torch.logsumexp(logits.float().masked_fill(~allowed,-torch.inf),dim=1) -
            torch.logsumexp(logits.float().masked_fill(~positive,-torch.inf),dim=1)).mean()

def model_identity(path):
    p = Path(path)
    files = sorted(list(p.glob("*.safetensors")) + list(p.glob("pytorch_model*.bin")))
    if not files:
        raise FileNotFoundError(f"No model weights at {p}")
    return digest({f.name:sha(f) for f in files} | {"config":sha(p/"config.json")})

def train(c, resume=False):
    import contextlib
    import torch
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(c["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["seed"])
    records = [json.loads(s) for s in (ROOT/"data/train_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    base = read_json(ROOT/"models/base.json")["path"]
    model, tokenizer = load_encoder(c,base)
    if c["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optimizer = torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    batches = math.ceil(len(records)/c["groups_per_batch"])
    steps_per_epoch = math.ceil(batches/c["grad_accum"])
    scheduler = get_linear_schedule_with_warmup(optimizer,int(steps_per_epoch*c["epochs"]*.1),steps_per_epoch*c["epochs"])
    amp = c["amp"] and device(c).type == "cuda"
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    resume_file = ROOT/"models/resume.pt"
    start_epoch, start_batch, global_step = 0,0,0
    if resume and resume_file.exists():
        state = torch.load(resume_file,map_location=device(c),weights_only=False)
        assert state["protocol_sha256"] == sha(ROOT/"protocol.json")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch,start_batch,global_step = state["epoch"],state["batch"],state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    def checkpoint(epoch,batch):
        temp = resume_file.with_suffix(".tmp")
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "scaler":scaler.state_dict(),"epoch":epoch,"batch":batch,"step":global_step,
                    "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if amp else [],
                    "protocol_sha256":sha(ROOT/"protocol.json")},temp)
        os.replace(temp,resume_file)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,c["epochs"]):
        order = list(records)
        random.Random(c["seed"]+epoch).shuffle(order)
        for b in range(start_batch if epoch==start_epoch else 0,batches):
            batch = order[b*c["groups_per_batch"]:(b+1)*c["groups_per_batch"]]
            unique = {d["doc_id"]:d for r in batch for d in [r["positive"]]+r["negatives"]}
            docs = list(unique.values())
            positive,allowed = contrastive_masks(batch,docs)
            ctx = torch.autocast("cuda",dtype=torch.float16) if amp else contextlib.nullcontext()
            with ctx:
                q = embeddings(model,tokenizer,[r["query"] for r in batch],c)
                p = embeddings(model,tokenizer,[d["input"] for d in docs],c)
                loss = multi_positive_loss(q @ p.T / c["temperature"],positive,allowed)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch} batch {b}")
            # Final accumulation window uses its actual size.
            window = min(c["grad_accum"],batches-(b//c["grad_accum"])*c["grad_accum"])
            scaler.scale(loss/window).backward()
            if (b+1)%c["grad_accum"]==0 or b+1==batches:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                log = {"epoch":epoch+1,"batch":b+1,"step":global_step,"loss_last_microbatch":loss.item(),
                       "grad_norm":float(norm),"learning_rate":scheduler.get_last_lr()[0],
                       "elapsed_seconds":time.perf_counter()-started}
                with open(ROOT/"logs/train.jsonl","a",encoding="utf-8") as f:
                    f.write(json.dumps(log)+"\n")
                if global_step%10==0:
                    print(json.dumps(log),flush=True)
                if global_step%50==0:
                    checkpoint(epoch,b+1)
        out = ROOT/f"models/epoch_{epoch+1}"
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        save_json(out/"experiment_provenance.json",{"base":read_json(ROOT/"models/base.json"),"epoch":epoch+1,
                  "protocol_sha256":sha(ROOT/"protocol.json"),"samples_sha256":sha(ROOT/"data/train_samples.jsonl"),
                  "training_config":c,"selection":"Not selected yet; full-corpus dev retrieval required."})
        checkpoint(epoch+1,0)
        start_batch = 0
    save_json(ROOT/"models/training_complete.json",{"epochs":c["epochs"],"steps":global_step,"selected":False})
