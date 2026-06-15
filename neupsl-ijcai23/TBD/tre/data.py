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
# 1. Helper: tìm token spaCy theo char
# ==============================

def get_token_for_char(tokens, char_idx):
    """
    tokens: spaCy Doc (hoặc list token có thuộc tính .idx)
    char_idx: vị trí ký tự trong text gốc

    Trả về:
        (token_index, token) sao cho:
        - nếu char nằm đúng tại token.idx -> token đó
        - nếu char nằm giữa 2 token -> token phía trước
        - nếu vượt cuối -> token cuối cùng
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
# 2. Cắt window context theo token
# ==============================

def get_context_by_window(text, doc, e1_start, e1_end, e2_start, e2_end, ws=None):
    """
    text: văn bản gốc (string)
    doc : spaCy Doc (đã tokenize)
    e1_start, e1_end, e2_start, e2_end: char offset của E1, E2 trên text gốc
    ws  : window size theo SỐ TOKEN (spaCy). Nếu None -> giữ nguyên (không cắt).

    Trả về:
        context_text,
        (e1_start_ctx, e1_end_ctx, e2_start_ctx, e2_end_ctx) theo context_text
    """
    tokens = list(doc)
    num_tokens = len(tokens)

    if ws is None:
        ws = num_tokens

    # khoảng bao trùm entity theo char
    start = min(e1_start, e2_start)
    end = max(e1_end, e2_end)

    start_token, _ = get_token_for_char(tokens, start)
    end_token, _ = get_token_for_char(tokens, end)

    # Nếu 2 entity cách nhau quá xa theo số token
    if end_token - start_token > ws:
        window = ws // 4  # mỗi bên lấy ws/4 quanh entity

        # Window cho E1 (thực ra là entity đứng trước)
        start_1_token = max(0, start_token - window)
        end_1_token = min(start_token + window, num_tokens - 1)

        # Window cho E2 (entity đứng sau)
        start_2_token = max(0, end_token - window)
        end_2_token = min(end_token + window, num_tokens - 1)

        start_1 = tokens[start_1_token].idx
        end_1 = tokens[end_1_token].idx + len(tokens[end_1_token])

        start_2 = tokens[start_2_token].idx
        end_2 = tokens[end_2_token].idx + len(tokens[end_2_token])

        text_1 = text[start_1:end_1]
        text_2 = text[start_2:end_2]

        # phần giữa bị bỏ (chỉ còn làm tham chiếu để chỉnh offset)
        mid = text[end_1:start_2]

        # context ghép 2 đoạn với newline
        context = "\n".join([text_1, text_2])

        # Điều chỉnh offset E1/E2 sang context
        # Case 1: (e1 nằm trong [start_1, end_1]) & (e2 trong [start_2, end_2])
        if start_1 <= e1_start and e1_end <= end_1:
            # E1 trong đoạn 1
            e1_start_ctx = e1_start - start_1
            e1_end_ctx = e1_end - start_1

            # E2 trong đoạn 2
            # context = text_1 + "\n" + text_2
            # độ dài phần nối giữa start_1 và start_2 trong text gốc: mid
            # nhưng trong context, đoạn 2 bắt đầu tại len(text_1) + 1
            # ta giữ cách tính gần với code gốc:
            shift_2 = (end_1 - start_1) + 1  # len(text_1) + 1 cho "\n"
            e2_start_ctx = shift_2 + (e2_start - start_2)
            e2_end_ctx = shift_2 + (e2_end - start_2)
        else:
            # Case 2: E1 ở đoạn 2, E2 ở đoạn 1
            # Hoán đổi logic (giữ đúng thứ tự index)
            e2_start_ctx = e2_start - start_1
            e2_end_ctx = e2_end - start_1

            shift_1 = (end_1 - start_1) + 1
            e1_start_ctx = shift_1 + (e1_start - start_2)
            e1_end_ctx = shift_1 + (e1_end - start_2)

        return context, (e1_start_ctx, e1_end_ctx, e2_start_ctx, e2_end_ctx)

    else:
        # 2 entity đủ gần, cắt 1 window bao chung
        # đảm bảo window không nhỏ hơn khoảng entity
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
# 3. Chèn tag [E1], [E2]
# ==============================

def add_event_tokens(text, span):
    """
    text: context đã cắt
    span: (event1_start, event1_end, event2_start, event2_end) theo context
    """
    event1_start, event1_end, event2_start, event2_end = span

    tag_start1, tag_start2 = " [E1] ", " [E2] "
    tag_end1, tag_end2 = " [/E1] ", " [/E2] "

    # Đảm bảo event1 là entity đứng trước trong context
    if event1_start > event2_start and event1_end > event2_start:
        event1_start, event1_end, event2_start, event2_end = (
            event2_start, event2_end, event1_start, event1_end
        )
        tag_start1, tag_start2 = tag_start2, tag_start1
        tag_end1, tag_end2 = tag_end2, tag_end1

    # Chèn từ phải sang trái để không làm lệch offset phía trước
    text = text[:event2_end] + tag_end2 + text[event2_end:]
    text = text[:event2_start] + tag_start2 + text[event2_start:]
    text = text[:event1_end] + tag_end1 + text[event1_end:]
    text = text[:event1_start] + tag_start1 + text[event1_start:]

    return text


# ==============================
# 4. Tạo marked_text cho từng sample
# ==============================

def create_marked_text_from_doc(
    text, doc, e1_start, e1_end, e2_start, e2_end,
    e1_text, e2_text, document_id, ws
):
    """
    Tạo context window quanh 2 entity, chèn [E1]/[E2],
    và kiểm tra lại bằng string để đảm bảo entity đúng.
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
# 5. Chuyển input_ids -> word_pieces
# ==============================

def group_words_fast(enc, tokenizer):
    """
    enc: output từ tokenizer (batch)
    tokenizer: HuggingFace tokenizer

    Trả về:
        word_pieces_all: list[B] mỗi phần tử là list token string
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
                # Xử lý prefix roberta/phobert
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
# 6. spaCy trên marked_text để lấy từ + vị trí marker
# ==============================

def process_docs_spacy_all(docs_spacy_all):
    """
    docs_spacy_all: list spaCy Doc, mỗi doc là 1 marked_text (đã có [E1]/[E2])

    Trả về:
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
            # bỏ token space rỗng
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
# 7. Tạo word_marks, e1_marks, e2_marks (dùng spacy_alignments)
# ==============================

def mark(
    word_pieces_all,
    words_text_all,
    e1s_ids_all,
    e1e_ids_all,
    e2s_ids_all,
    e2e_ids_all,
    B,  # batch size
    M,  # max số word (spaCy)
    L,  # max số wordpiece
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
        # ALIGN BERT subword ↔ spaCy word bằng spacy_alignments
        # a2b[sub_idx] = list[word_idx]
        # b2a[word_idx] = list[sub_idx]
        a2b, b2a = tokenizations.get_alignments(wp, wt)

        # word_marks: đánh cho dữ liệu word-level (trừ marker)
        for w_idx, sub_ids in enumerate(b2a):
            if w_idx >= M:
                break
            if wt[w_idx] in ("[E1]", "[/E1]", "[E2]", "[/E2]"):
                continue
            if not sub_ids:
                continue
            word_marks[i, w_idx, sub_ids] = 1

        # e1_marks: union subword của từ nằm giữa [E1] và [/E1]
        e1_ids = [idx for ids in b2a[e1s + 1 : e1e] for idx in ids]
        if e1_ids:
            e1_marks[i, 0, e1_ids] = 1

        # e2_marks: union subword của từ nằm giữa [E2] và [/E2]
        e2_ids = [idx for ids in b2a[e2s + 1 : e2e] for idx in ids]
        if e2_ids:
            e2_marks[i, 0, e2_ids] = 1

    return word_marks, e1_marks, e2_marks


# ==============================
# 8. Hàm build cache: spaCy + token + mark
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
    df: DataFrame có các cột:
        - text
        - label_encoded
        - entity1_start, entity1_end, entity2_start, entity2_end
        - entity1_text, entity2_text
        - entity1_id, entity2_id, document_id
        - entity1_type, entity2_type   <-- MỚI
    """
    TYPE2ID = {
        'UNKNOWN': 0, 'OCCURRENCE': 1, 'TREATMENT': 2, 'TEST': 3, 'DURATION': 4,
        'PROBLEM': 5, 'CLINICAL_DEPT': 6, 'EVIDENTIAL': 7, 'DATE': 8, 'FREQUENCY': 9, 'TIME': 10
    }
    t0 = time.time()

    # Cố định uid
    df = df.reset_index(drop=False).rename(columns={"index": "uid"})

    uids = df["uid"].tolist()
    texts = df["text"].tolist()
    labels_all = torch.tensor(df["label_encoded"].tolist(), dtype=torch.long)

    # 1) spaCy trên raw text (để lấy char offset chính xác)
    print("spaCy processing raw texts...")
    docs_raw = list(spacy_nlp.pipe(texts, batch_size=batch_size, n_process=n_process))
    print("Done spaCy on raw texts.")

    # 2) Tạo marked_texts + filter sample lỗi
    print("Creating marked texts...")
    kept_uids = []
    kept_labels = []
    kept_marked_texts = []
    kept_e1_ids = []
    kept_e2_ids = []
    kept_doc_ids = []
    kept_e1_type_ids = []   # id đã mapping
    kept_e2_type_ids = []   # id đã mapping

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
            # bỏ sample này, tránh phá align
            continue

        kept_uids.append(int(uid))
        kept_labels.append(label)
        kept_marked_texts.append(marked)
        kept_e1_ids.append(ent1_id)
        kept_e2_ids.append(ent2_id)
        kept_doc_ids.append(docid)

        # --- mapping type string -> id trước khi lưu ---
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

    # 4) Chuyển input_ids -> word_pieces
    word_pieces_all = group_words_fast(enc, tokenizer)

    # 5) spaCy trên marked_text (lần 2)
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

    # 6) Tạo word_marks, e1_marks, e2_marks
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

    # 7) Ghi jsonl metadata (lưu thêm type string cho dễ debug, optional)
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
        "type2id": TYPE2ID,   # lưu lại mapping để reference
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
    Load cache sinh bởi build_spacy_and_token_cache.

    ĐẢM BẢO:
      1) Thứ tự ban đầu được khôi phục theo uid_order (index trong .pt)
      2) Sau đó SẮP XẾP global theo doc_ids (sorted by doc_id)
    """

    def _jsonl_iter(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ===================== 1. Load từ .pt =====================
    data = torch.load(token_pt_path, map_location="cpu", weights_only=False)

    input_ids      = data["input_ids"]              # [N,L]
    attention_mask = data["attention_mask"]         # [N,L]
    e1_marks       = data["e1_marks"]               # [N,1,L] hoặc [N,L]
    e2_marks       = data["e2_marks"]               # [N,1,L] hoặc [N,L]
    word_marks     = data.get("word_marks", None)   # [N,Wmax,L]
    labels         = data.get("labels", None)       # [N]
    indices        = data.get("index", None)        # [N]
    metadata       = data.get("metadata", {})
    e1_type_ids       = data.get("e1_type_ids", None)     # [N] hoặc None (nếu cache cũ)
    e2_type_ids       = data.get("e2_type_ids", None)     # [N] hoặc None

    N = input_ids.size(0)

    # ===================== 2. Load JSONL =====================
    uid2rec = {}
    for obj in _jsonl_iter(jsonl_path):
        uid2rec[int(obj["uid"])] = obj

    # ===================== 3. Lấy thứ tự uid theo .pt =====================
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
        "e1_type_ids":       e1_type_ids,   # [N] hoặc None
        "e2_type_ids":       e2_type_ids,   # [N] hoặc None
        "e1_ids":         e1_ids,
        "e2_ids":         e2_ids,
        "doc_ids":        doc_ids,
        "words_spacy":    words_spacy,
        "metadata":       metadata,
        "index":          uid_sorted,
    }


class TRECachedDataset(Dataset):
    """
    Dataset đọc từ cache .pt và JSONL.

    - Thứ tự global của mẫu đã được sort theo doc_id trong load_cached_dataset.
    """

    def __init__(self, token_cache_path, spacy_jsonl_path):
        cached = load_cached_dataset(token_cache_path, spacy_jsonl_path)

        self.input_ids   = cached["input_ids"]          # [N,L]
        self.attn_mask   = cached["attention_mask"]     # [N,L]
        self.e1_marks    = cached["e1_marks"]           # [N,1,L] hoặc [N,L]
        self.e2_marks    = cached["e2_marks"]           # [N,1,L] hoặc [N,L]
        self.word_marks  = cached["word_marks"]         # [N,Wmax,L]
        self.labels      = cached["labels"]             # [N] (có thể None)
        self.e1_type_ids    = cached["e1_type_ids"]           # [N] (id), có thể None nếu cache cũ
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
      - Sort lại các sample trong batch theo doc_ids.
      - Sau đó stack tensor.
    """
    # sort batch theo doc_ids trong batch
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
    Tạo DataLoader:

    - Dataset đã sort global theo doc_id.
    - Mỗi batch khi collate sẽ sort lại theo doc_id trong batch.
    """
    ds = TRECachedDataset(token_cache_path, spacy_jsonl_path)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,          # nếu muốn random doc thì bật True
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
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Ánh xạ giống tập huấn luyện

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
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])  # Ánh xạ giống tập huấn luyện
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
