import os
import time
import torch
import orjson
import spacy_alignments as tokenizations
import pandas as pd

import json
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# ==============================
# 1. Helper: find the spaCy token for a character offset
# ==============================

def get_token_for_char(tokens, char_idx):
    """
    tokens: spaCy Doc (or a token list with an .idx attribute)
    char_idx: character offset in the original text

    Returns:
        (token_index, token) such that:
        - if the character lands exactly on token.idx -> that token
        - if the character falls between two tokens -> the previous token
        - if it is past the end -> the last token
    """
    for i, token in enumerate(tokens):
        if char_idx > token.idx:
            continue
        if char_idx == token.idx:
            return i, token
        if char_idx < token.idx:
            return i - 1, tokens[i - 1]
    return len(tokens) - 1, tokens[len(tokens) - 1]


# ==============================
# 2. Slice a token-based context window
# ==============================

def get_context_by_window(text, doc, e1_start, e1_end, e2_start, e2_end, ws=None):
    """
    text: original text (string)
    doc : spaCy Doc (already tokenized)
    e1_start, e1_end, e2_start, e2_end: character offsets of E1 and E2 in the original text
    ws  : window size in TOKEN count (spaCy). If None -> keep the full text (no trimming).

    Returns:
        context_text,
        (e1_start_ctx, e1_end_ctx, e2_start_ctx, e2_end_ctx) theo context_text
    """
    tokens = list(doc)
    num_tokens = len(tokens)

    if ws is None:
        ws = num_tokens

    # character span covering the entities
    start = min(e1_start, e2_start)
    end = max(e1_end, e2_end)

    start_token, _ = get_token_for_char(tokens, start)
    end_token, _ = get_token_for_char(tokens, end)

    # If the two entities are too far apart in token count
    if end_token - start_token > ws:
        window = ws // 4  # take ws/4 around each entity

        # Window for E1 (the earlier entity)
        start_1_token = max(0, start_token - window)
        end_1_token = min(start_token + window, num_tokens - 1)

        # Window for E2 (the later entity)
        start_2_token = max(0, end_token - window)
        end_2_token = min(end_token + window, num_tokens - 1)

        start_1 = tokens[start_1_token].idx
        end_1 = tokens[end_1_token].idx + len(tokens[end_1_token])

        start_2 = tokens[start_2_token].idx
        end_2 = tokens[end_2_token].idx + len(tokens[end_2_token])

        text_1 = text[start_1:end_1]
        text_2 = text[start_2:end_2]

        # the middle segment is dropped (kept only as a reference for offset adjustment)
        mid = text[end_1:start_2]

        # context joins the two spans with a newline
        context = "\n".join([text_1, text_2])

        # Adjust E1/E2 offsets into the context
        # Case 1: (e1 is inside [start_1, end_1]) and (e2 is inside [start_2, end_2])
        if start_1 <= e1_start and e1_end <= end_1:
            # E1 is in segment 1
            e1_start_ctx = e1_start - start_1
            e1_end_ctx = e1_end - start_1

            # E2 is in segment 2
            # context = text_1 + "\n" + text_2
            # length of the gap between start_1 and start_2 in the original text: mid
            # but in the context, segment 2 starts at len(text_1) + 1
            # keep the computation close to the original code:
            shift_2 = (end_1 - start_1) + 1  # len(text_1) + 1 for "\n"
            e2_start_ctx = shift_2 + (e2_start - start_2)
            e2_end_ctx = shift_2 + (e2_end - start_2)
        else:
            # Case 2: E1 is in segment 2, E2 is in segment 1
            # Swap the logic while preserving index order
            e2_start_ctx = e2_start - start_1
            e2_end_ctx = e2_end - start_1

            shift_1 = (end_1 - start_1) + 1
            e1_start_ctx = shift_1 + (e1_start - start_2)
            e1_end_ctx = shift_1 + (e1_end - start_2)

        return context, (e1_start_ctx, e1_end_ctx, e2_start_ctx, e2_end_ctx)

    else:
        # The two entities are close enough, so slice one shared window
        # ensure the window is not smaller than the entity span
        window = max(0, (ws - (end_token - start_token)) // 2)

        start_token_ctx = max(0, start_token - window)
        end_token_ctx = min(num_tokens - 1, end_token + window)

        start_ctx = tokens[start_token_ctx].idx
        end_ctx = tokens[end_token_ctx].idx + len(tokens[end_token_ctx])

        context = text[start_ctx:end_ctx]

        e1_start_ctx = e1_start - start_ctx
        e1_end_ctx = e1_end - start_ctx
        e2_start_ctx = e2_start - start_ctx
        e2_end_ctx = e2_end - start_ctx

        return context, (e1_start_ctx, e1_end_ctx, e2_start_ctx, e2_end_ctx)


# ==============================
# 3. Insert the [E1], [E2] tags
# ==============================

def add_event_tokens(text, span):
    """
    text: trimmed context
    span: (event1_start, event1_end, event2_start, event2_end) in the context
    """
    event1_start, event1_end, event2_start, event2_end = span

    tag_start1, tag_start2 = " [E1] ", " [E2] "
    tag_end1, tag_end2 = " [/E1] ", " [/E2] "

    # Ensure event1 is the earlier entity in the context
    if event1_start > event2_start and event1_end > event2_start:
        event1_start, event1_end, event2_start, event2_end = (
            event2_start, event2_end, event1_start, event1_end
        )
        tag_start1, tag_start2 = tag_start2, tag_start1
        tag_end1, tag_end2 = tag_end2, tag_end1

    # Insert from right to left so earlier offsets are not shifted
    text = text[:event2_end] + tag_end2 + text[event2_end:]
    text = text[:event2_start] + tag_start2 + text[event2_start:]
    text = text[:event1_end] + tag_end1 + text[event1_end:]
    text = text[:event1_start] + tag_start1 + text[event1_start:]

    return text


# ==============================
# 4. Build marked_text for each sample
# ==============================

def create_marked_text_from_doc(
    text, doc, e1_start, e1_end, e2_start, e2_end,
    e1_text, e2_text, document_id, ws
):
    """
    Build a context window around the two entities, insert [E1]/[E2],
    and verify with string matching that the entities are correct.
    """
    context, span = get_context_by_window(
        text, doc, e1_start, e1_end, e2_start, e2_end, ws
    )

    marked_text = add_event_tokens(context, span)
    marked_text = marked_text.replace("\n", " ").replace("  ", " ")

    marked_e1 = f"[E1] {e1_text} [/E1]".replace("\n", " ").replace("  ", " ")
    marked_e2 = f"[E2] {e2_text} [/E2]".replace("\n", " ").replace("  ", " ")

    if (marked_e1 in marked_text) and (marked_e2 in marked_text):
        return marked_text
    else:
        print(f"[Warning] Missing markers for doc_id={document_id}")
        return "Error"


# ==============================
# 5. Convert input_ids -> word_pieces
# ==============================

def group_words_fast(enc, tokenizer):
    """
    enc: tokenizer output (batch)
    tokenizer: HuggingFace tokenizer

    Returns:
        word_pieces_all: list[B], each element is a list of token strings
    """
    input_ids = enc["input_ids"]
    B, L = input_ids.shape

    is_roberta = (
        ("roberta" in getattr(tokenizer, "name_or_path", "").lower()) or
        ("phobert" in getattr(tokenizer, "name_or_path", "").lower())
    )

    word_pieces_all = []
    for i in range(B):
        word_pieces = []
        for idx in input_ids[i]:
            wp = tokenizer.convert_ids_to_tokens(idx.item())
            if wp in ("[PAD]", "<pad>"):
                break
            if is_roberta:
                # Handle the roberta/phobert prefix
                wp = (
                    wp.replace("Ġ", " ")
                      .replace("Ċ", "\n")
                      .replace("ĉ", "")
                      .strip()
                )
            else:
                wp = wp.replace("##", "")
            word_pieces.append(wp)
        word_pieces_all.append(word_pieces)

    return word_pieces_all


# ==============================
# 6. Run spaCy on marked_text to extract words and marker positions
# ==============================

def process_docs_spacy_all(docs_spacy_all):
    """
    docs_spacy_all: list of spaCy Docs, each one is a marked_text (already containing [E1]/[E2])

    Returns:
        words_spacy_all: [B][num_words] dict token info
        words_text_all : [B][num_words] string token
        e1s_ids_all    : [B] index token "[E1]"
        e1e_ids_all    : [B] index token "[/E1]"
        e2s_ids_all    : [B] index token "[E2]"
        e2e_ids_all    : [B] index token "[/E2]"
    """
    words_spacy_all = []
    words_text_all = []
    e1s_ids_all = []
    e1e_ids_all = []
    e2s_ids_all = []
    e2e_ids_all = []

    for doc_spacy in docs_spacy_all:
        doc_clean = []
        for tok in doc_spacy:
            # drop empty space tokens
            if tok.text.strip() == "":
                continue
            doc_clean.append({
                "i": tok.i,
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "morph": tok.morph.to_dict(),
                "dep": tok.dep_,
                "span": [tok.idx, tok.idx + len(tok)]
            })

        words_text = [w["text"] for w in doc_clean]

        try:
            e1s_idx = words_text.index("[E1]")
            e1e_idx = words_text.index("[/E1]")
            e2s_idx = words_text.index("[E2]")
            e2e_idx = words_text.index("[/E2]")
        except ValueError:
            print("\nERROR: Marker missing in marked_text")
            print(words_text)
            raise

        words_spacy_all.append(doc_clean)
        words_text_all.append(words_text)
        e1s_ids_all.append(e1s_idx)
        e1e_ids_all.append(e1e_idx)
        e2s_ids_all.append(e2s_idx)
        e2e_ids_all.append(e2e_idx)

    return (
        words_spacy_all,
        words_text_all,
        e1s_ids_all,
        e1e_ids_all,
        e2s_ids_all,
        e2e_ids_all,
    )


# ==============================
# 7. Build word_marks, e1_marks, e2_marks (using spacy_alignments)
# ==============================

def mark(
    word_pieces_all,
    words_text_all,
    e1s_ids_all,
    e1e_ids_all,
    e2s_ids_all,
    e2e_ids_all,
    B,  # batch size
    M,  # max number of words (spaCy)
    L,  # max number of wordpieces
):
    """
    word_pieces_all: [B][T_subword]
    words_text_all : [B][T_word]
    e1/e2*_ids_all : [B] index marker trong words_text_all
    """

    word_marks = torch.zeros((B, M, L))
    e1_marks = torch.zeros((B, 1, L))
    e2_marks = torch.zeros((B, 1, L))

    for i, (wp, wt, e1s, e1e, e2s, e2e) in enumerate(
        zip(
            word_pieces_all,
            words_text_all,
            e1s_ids_all,
            e1e_ids_all,
            e2s_ids_all,
            e2e_ids_all,
        )
    ):
        # Align BERT subwords ↔ spaCy words using spacy_alignments
        # a2b[sub_idx] = list[word_idx]
        # b2a[word_idx] = list[sub_idx]
        a2b, b2a = tokenizations.get_alignments(wp, wt)

        # word_marks: mark the word-level data (excluding markers)
        for w_idx, sub_ids in enumerate(b2a):
            if w_idx >= M:
                break
            if wt[w_idx] in ("[E1]", "[/E1]", "[E2]", "[/E2]"):
                continue
            if not sub_ids:
                continue
            word_marks[i, w_idx, sub_ids] = 1

        # e1_marks: union of subwords for words inside [E1] and [/E1]
        e1_ids = [idx for ids in b2a[e1s + 1 : e1e] for idx in ids]
        if e1_ids:
            e1_marks[i, 0, e1_ids] = 1

        # e2_marks: union of subwords for words inside [E2] and [/E2]
        e2_ids = [idx for ids in b2a[e2s + 1 : e2e] for idx in ids]
        if e2_ids:
            e2_marks[i, 0, e2_ids] = 1

    return word_marks, e1_marks, e2_marks


# ==============================
# 8. Cache builder: spaCy + token + marks
# ==============================

def build_spacy_and_token_cache(
    df,
    spacy_nlp,
    tokenizer,
    window_size,
    max_length,
    jsonl_path,
    token_pt,
    batch_size=256,
    n_process=1,
):
    """
    df: DataFrame with the following columns:
        - text
        - label_encoded
        - entity1_start, entity1_end, entity2_start, entity2_end
        - entity1_text, entity2_text
        - entity1_id, entity2_id, document_id
        - entity1_type, entity2_type   <-- NEW
    """
    TYPE2ID = {
        'UNKNOWN': 0, 'OCCURRENCE': 1, 'TREATMENT': 2, 'TEST': 3, 'DURATION': 4,
        'PROBLEM': 5, 'CLINICAL_DEPT': 6, 'EVIDENTIAL': 7, 'DATE': 8, 'FREQUENCY': 9, 'TIME': 10
    }
    t0 = time.time()

    # Fix the uid
    df = df.reset_index(drop=False).rename(columns={"index": "uid"})

    uids = df["uid"].tolist()
    texts = df["text"].tolist()
    labels_all = torch.tensor(df["label_encoded"].tolist(), dtype=torch.long)

    # 1) Run spaCy on the raw text (to obtain exact character offsets)
    print("spaCy processing raw texts...")
    docs_raw = list(spacy_nlp.pipe(texts, batch_size=batch_size, n_process=n_process))
    print("Done spaCy on raw texts.")

    # 2) Build marked_texts + filter out invalid samples
    print("Creating marked texts...")
    kept_uids = []
    kept_labels = []
    kept_marked_texts = []
    kept_e1_ids = []
    kept_e2_ids = []
    kept_doc_ids = []
    kept_e1_type_ids = []   # mapped ids
    kept_e2_type_ids = []   # mapped ids

    for (
        uid,
        text,
        doc_raw,
        e1s,
        e1e,
        e2s,
        e2e,
        e1t,
        e2t,
        docid,
        label,
        ent1_id,
        ent2_id,
        ent1_type_str,
        ent2_type_str,
    ) in zip(
        uids,
        texts,
        docs_raw,
        df["entity1_start"],
        df["entity1_end"],
        df["entity2_start"],
        df["entity2_end"],
        df["entity1_text"],
        df["entity2_text"],
        df["document_id"],
        labels_all,
        df["entity1_id"],
        df["entity2_id"],
        df["entity1_type"],
        df["entity2_type"],
    ):
        marked = create_marked_text_from_doc(
            text,
            doc_raw,
            int(e1s),
            int(e1e),
            int(e2s),
            int(e2e),
            e1t,
            e2t,
            docid,
            ws=window_size,
        )

        if marked == "Error":
            # drop this sample to avoid breaking alignment
            continue

        kept_uids.append(int(uid))
        kept_labels.append(label)
        kept_marked_texts.append(marked)
        kept_e1_ids.append(ent1_id)
        kept_e2_ids.append(ent2_id)
        kept_doc_ids.append(docid)

        # --- map type strings to ids before saving ---
        t1 = str(ent1_type_str) if ent1_type_str is not None else "UNKNOWN"
        t2 = str(ent2_type_str) if ent2_type_str is not None else "UNKNOWN"
        kept_e1_type_ids.append(TYPE2ID.get(t1, TYPE2ID["UNKNOWN"]))
        kept_e2_type_ids.append(TYPE2ID.get(t2, TYPE2ID["UNKNOWN"]))

    if len(kept_marked_texts) == 0:
        raise RuntimeError("No valid samples after create_marked_text_from_doc.")

    labels = torch.stack(kept_labels, dim=0)
    e1_type_ids = torch.tensor(kept_e1_type_ids, dtype=torch.long)  # [N]
    e2_type_ids = torch.tensor(kept_e2_type_ids, dtype=torch.long)  # [N]

    print(f"Created {len(kept_marked_texts)} marked texts (after filtering).")

    # 3) Tokenize batch
    enc = tokenizer(
        kept_marked_texts,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # 4) Convert input_ids -> word_pieces
    word_pieces_all = group_words_fast(enc, tokenizer)

    # 5) Run spaCy on marked_text (second pass)
    print("spaCy processing marked texts...")
    docs_spacy_all = list(
        spacy_nlp.pipe(
            kept_marked_texts, batch_size=batch_size, n_process=n_process
        )
    )
    print("Done spaCy on marked texts.")

    (
        words_spacy_all,
        words_text_all,
        e1s_ids_all,
        e1e_ids_all,
        e2s_ids_all,
        e2e_ids_all,
    ) = process_docs_spacy_all(docs_spacy_all)

    # 6) Build word_marks, e1_marks, e2_marks
    B, L = enc["input_ids"].shape
    M = max(len(ws) for ws in words_spacy_all)

    word_marks, e1_marks, e2_marks = mark(
        word_pieces_all,
        words_text_all,
        e1s_ids_all,
        e1e_ids_all,
        e2s_ids_all,
        e2e_ids_all,
        B,
        M,
        L,
    )

    # 7) Write jsonl metadata (store type strings as an optional debug aid)
    with open(jsonl_path, "wb") as f:
        for uid, e1_id, e2_id, doc_id, marked_text, words_by_spacy, e1_type_id, e2_type_id in zip(
            kept_uids,
            kept_e1_ids,
            kept_e2_ids,
            kept_doc_ids,
            kept_marked_texts,
            words_spacy_all,
            kept_e1_type_ids,
            kept_e2_type_ids,
        ):
            rec = {
                "uid": int(uid),
                "e1_id": e1_id,
                "e2_id": e2_id,
                "doc_id": doc_id,
                "marked_text": marked_text,
                "words_by_spacy": words_by_spacy,
                "e1_type_id": int(e1_type_id),
                "e2_type_id": int(e2_type_id),
            }
            f.write(orjson.dumps(rec) + b"\n")

    # 8) Save token cache (.pt) + metadata
    metadata = {
        "tokenizer_name": getattr(tokenizer, "name_or_path", str(type(tokenizer))),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "max_length": max_length,
        "window_size": window_size,
        "num_samples": len(kept_uids),
        "spacy_pipeline": [p for p, _ in spacy_nlp.pipeline],
        "type2id": TYPE2ID,   # keep the mapping for reference
    }

    torch.save(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "word_marks": word_marks,
            "e1_marks": e1_marks,
            "e2_marks": e2_marks,
            "labels": labels,
            "e1_type_ids": e1_type_ids,  # [N]
            "e2_type_ids": e2_type_ids,  # [N]
            "index": kept_uids,
            "metadata": metadata,
        },
        token_pt,
    )

    print(f"Saved cache:\n - {jsonl_path}\n - {token_pt}")
    print(f"Done in {time.time() - t0:.2f}s")



def load_cached_dataset(token_pt_path, jsonl_path):
    """
    Load the cache generated by build_spacy_and_token_cache.

    ENSURE:
      1) The original order is restored via uid_order (the index in the .pt file)
      2) Then globally sort by doc_ids (sorted by doc_id)
    """

    def _jsonl_iter(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ===================== 1. Load from .pt =====================
    data = torch.load(token_pt_path, map_location="cpu", weights_only=False)

    input_ids      = data["input_ids"]              # [N,L]
    attention_mask = data["attention_mask"]         # [N,L]
    e1_marks       = data["e1_marks"]               # [N,1,L] or [N,L]
    e2_marks       = data["e2_marks"]               # [N,1,L] or [N,L]
    word_marks     = data.get("word_marks", None)   # [N,Wmax,L]
    labels         = data.get("labels", None)       # [N]
    indices        = data.get("index", None)        # [N]
    metadata       = data.get("metadata", {})
    e1_type_ids       = data.get("e1_type_ids", None)     # [N] or None (if using an older cache)
    e2_type_ids       = data.get("e2_type_ids", None)     # [N] or None

    N = input_ids.size(0)

    # ===================== 2. Load JSONL =====================
    uid2rec = {}
    for obj in _jsonl_iter(jsonl_path):
        uid2rec[int(obj["uid"])] = obj

    # ===================== 3. Recover uid order from the .pt file =====================
    if indices is not None:
        if torch.is_tensor(indices):
            uid_order = indices.tolist()
        else:
            uid_order = list(indices)
    else:
        uid_order = sorted(uid2rec.keys())

    if len(uid_order) != N:
        raise ValueError(
            f"Mismatch .pt and .jsonl size: pt={N}, jsonl_uids={len(uid_order)}"
        )

    # ===================== 4. Reorder theo uid_order =====================
    e1_ids      = [uid2rec[uid]["e1_id"] for uid in uid_order]
    e2_ids      = [uid2rec[uid]["e2_id"] for uid in uid_order]
    doc_ids     = [uid2rec[uid]["doc_id"] for uid in uid_order]
    words_spacy = [uid2rec[uid]["words_by_spacy"] for uid in uid_order]

    # ===================== 5. SORTED global theo doc_ids =====================
    sorted_idx = sorted(range(N), key=lambda i: str(doc_ids[i]))

    # reorder tensor
    input_ids      = input_ids[sorted_idx]
    attention_mask = attention_mask[sorted_idx]
    e1_marks       = e1_marks[sorted_idx]
    e2_marks       = e2_marks[sorted_idx]
    if word_marks is not None:
        word_marks = word_marks[sorted_idx]
    if labels is not None:
        labels = labels[sorted_idx]
    if e1_type_ids is not None:
        e1_type_ids = e1_type_ids[sorted_idx]
    if e2_type_ids is not None:
        e2_type_ids = e2_type_ids[sorted_idx]

    # reorder list python
    e1_ids      = [e1_ids[i] for i in sorted_idx]
    e2_ids      = [e2_ids[i] for i in sorted_idx]
    doc_ids     = [doc_ids[i] for i in sorted_idx]
    words_spacy = [words_spacy[i] for i in sorted_idx]
    uid_sorted  = [uid_order[i] for i in sorted_idx]

    # ===================== 6. Return =====================
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "e1_marks":       e1_marks,
        "e2_marks":       e2_marks,
        "word_marks":     word_marks,
        "labels":         labels,
        "e1_type_ids":       e1_type_ids,   # [N] or None
        "e2_type_ids":       e2_type_ids,   # [N] or None
        "e1_ids":         e1_ids,
        "e2_ids":         e2_ids,
        "doc_ids":        doc_ids,
        "words_spacy":    words_spacy,
        "metadata":       metadata,
        "index":          uid_sorted,
    }


class TRECachedDataset(Dataset):
    """
    Dataset loaded from .pt and JSONL caches.

    - The global sample order has already been sorted by doc_id in load_cached_dataset.
    """

    def __init__(self, token_cache_path, spacy_jsonl_path):
        cached = load_cached_dataset(token_cache_path, spacy_jsonl_path)

        self.input_ids   = cached["input_ids"]          # [N,L]
        self.attn_mask   = cached["attention_mask"]     # [N,L]
        self.e1_marks    = cached["e1_marks"]           # [N,1,L] or [N,L]
        self.e2_marks    = cached["e2_marks"]           # [N,1,L] or [N,L]
        self.word_marks  = cached["word_marks"]         # [N,Wmax,L]
        self.labels      = cached["labels"]             # [N] (may be None)
        self.e1_type_ids    = cached["e1_type_ids"]           # [N] (id), may be None if using an older cache
        self.e2_type_ids    = cached["e2_type_ids"]           # [N]
        self.e1_ids      = cached["e1_ids"]             # list length N
        self.e2_ids      = cached["e2_ids"]             # list length N
        self.doc_ids     = cached["doc_ids"]            # list length N
        self.words_spacy = cached["words_spacy"]        # list length N
        self.index       = cached["index"]              # list uid


    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "e1_marks":       self.e1_marks[idx],
            "e2_marks":       self.e2_marks[idx],
            "word_marks":     self.word_marks[idx],
            "labels":         self.labels[idx] if self.labels is not None else -1,
            "e1_type_ids":       self.e1_type_ids[idx],   # scalar tensor
            "e2_type_ids":       self.e2_type_ids[idx],
            "e1_ids":         self.e1_ids[idx],
            "e2_ids":         self.e2_ids[idx],
            "doc_ids":        self.doc_ids[idx],
            "words_spacy":    self.words_spacy[idx],
            "uid":            self.index[idx],
        }

def tre_collate_cached(batch):
    """
    Collate function:
      - Re-sort the samples in the batch by doc_ids.
      - Then stack the tensors.
    """
    # Sort the batch by doc_ids.
    batch_sorted = sorted(batch, key=lambda b: str(b["doc_ids"]))

    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch_sorted]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch_sorted]),
        "e1_marks":       torch.stack([b["e1_marks"] for b in batch_sorted]),
        "e2_marks":       torch.stack([b["e2_marks"] for b in batch_sorted]),
        "word_marks":     torch.stack([b["word_marks"] for b in batch_sorted]),
        "labels":         torch.stack([b["labels"] for b in batch_sorted]),
        "e1_type_ids":       torch.stack([b["e1_type_ids"] for b in batch_sorted]),  # [B]
        "e2_type_ids":       torch.stack([b["e2_type_ids"] for b in batch_sorted]),  # [B]
        "e1_ids":         [b["e1_ids"] for b in batch_sorted],
        "e2_ids":         [b["e2_ids"] for b in batch_sorted],
        "doc_ids":        [b["doc_ids"] for b in batch_sorted],
        "words_spacy":    [b["words_spacy"] for b in batch_sorted],
        "uids":           [b["uid"] for b in batch_sorted],
    }



def create_dataloader(token_cache_path,
                      spacy_jsonl_path,
                      batch_size=32,
                      shuffle=False,
                      num_workers=0,
                      pin_memory=False):
    """
    Create the DataLoader:

    - The dataset is globally sorted by doc_id.
    - Each batch is re-sorted by doc_id during collation.
    """
    ds = TRECachedDataset(token_cache_path, spacy_jsonl_path)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,          # set True to randomize docs
        num_workers=num_workers,
        collate_fn=tre_collate_cached,
        pin_memory=pin_memory
    )


def loadi2b2():
    data_source = "/data/ddao/TRE/data/raw_data/i2b2_2012/data_processed/"
    data_train_path = data_source + "train_merged_full.csv"
    data_test_path = data_source + "test_merged_full.csv"
    
    
    train_df = pd.read_csv(data_train_path)
    test_df = pd.read_csv(data_test_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Use the same mapping as the training set

    num_classes = len(label_encoder.classes_)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    id2label = {v: k for k, v in label_mapping.items()}
    
    validation_size = 0.15
    random_state = 42
    train_df, valid_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
    train_df['type_encoded'] = label_encoder.fit_transform(train_df['label'])

    
    return {'train_df':train_df,
           'valid_df':valid_df,
           'test_df':test_df,
           'num_classes':num_classes,
           'label_mapping':label_mapping,
           'id2label':id2label}

def loadMATRES():
    data_source = "/data/ddao/TRE/data/raw_data/MATRES/data_processed/"
    data_train_path = data_source + "train.csv"
    data_test_path = data_source + "test.csv"
    
    train_df = pd.read_csv(data_train_path)
    test_df = pd.read_csv(data_test_path)
    
    # Add missing entity type columns
    train_df['entity1_type'] = 'UNKNOWN' 
    train_df['entity2_type'] = 'UNKNOWN'
    test_df['entity1_type'] = 'UNKNOWN'
    test_df['entity2_type'] = 'UNKNOWN'
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Use the same mapping as the training set
    num_classes = len(label_encoder.classes_)
    
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Label Mapping: {label_mapping}")
    relations = list(label_mapping.keys())
    id2label = {v: k for k, v in label_mapping.items()}
    
    validation_size = 0.15
    random_state = 42
    train_df, valid_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
    
    return {'train_df':train_df,
           'valid_df':valid_df,
           'test_df':test_df,
           'num_classes':num_classes,
           'label_mapping':label_mapping,
           'id2label':id2label}
