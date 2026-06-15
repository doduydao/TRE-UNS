
import random
import math
from collections import Counter
from torch.utils.data import Sampler


def _doc_sort_key(doc_id):
    s = str(doc_id)
    try:
        return (0, int(s))
    except (ValueError, TypeError):
        return (1, s)


class SortedDocSampler(Sampler):
    """
    Sample indices in deterministic order grouped by ascending doc_id.
    Keeps original relative order within each doc.
    """
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids
        self.ordered_indices = sorted(
            range(len(doc_ids)),
            key=lambda i: (_doc_sort_key(doc_ids[i]), i)
        )

    def __iter__(self):
        for idx in self.ordered_indices:
            yield idx

    def __len__(self):
        return len(self.ordered_indices)


class TripletAwarePairBatchSampler(Sampler):
    """
    Batch sampler for pair mode that tries to maximize in-batch transitivity triplets.

    Strategy:
    - Keep each batch within a single document.
    - Within a document, greedily pack pairs that share entities.
      This increases the chance of having (x,y), (y,z), (x,z) in one batch.
    """
    def __init__(self, doc_ids, e1_ids, e2_ids, batch_size, shuffle_docs=False):
        self.doc_ids = doc_ids
        self.e1_ids = e1_ids
        self.e2_ids = e2_ids
        self.batch_size = batch_size
        self.shuffle_docs = shuffle_docs

        self.doc_groups = {}
        for idx, doc_id in enumerate(self.doc_ids):
            d = str(doc_id)
            if d not in self.doc_groups:
                self.doc_groups[d] = []
            self.doc_groups[d].append(idx)

        self.doc_keys = sorted(self.doc_groups.keys(), key=_doc_sort_key)
        self.batches = self._make_batches()

    def _score_pair(self, idx, ent_freq):
        e1 = self.e1_ids[idx]
        e2 = self.e2_ids[idx]
        return ent_freq.get(e1, 0) + ent_freq.get(e2, 0)

    def _make_doc_chunks(self, indices):
        if len(indices) <= self.batch_size:
            return [indices]

        ent_freq = Counter()
        for idx in indices:
            ent_freq[self.e1_ids[idx]] += 1
            ent_freq[self.e2_ids[idx]] += 1

        remaining = set(indices)
        chunks = []

        while remaining:
            seed = max(remaining, key=lambda idx: self._score_pair(idx, ent_freq))
            chunk = [seed]
            remaining.remove(seed)

            chunk_entities = {self.e1_ids[seed], self.e2_ids[seed]}

            while remaining and len(chunk) < self.batch_size:
                best_idx = None
                best_key = None

                for idx in remaining:
                    e1 = self.e1_ids[idx]
                    e2 = self.e2_ids[idx]
                    shared = int(e1 in chunk_entities) + int(e2 in chunk_entities)
                    degree = self._score_pair(idx, ent_freq)
                    key = (shared, degree)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_idx = idx

                chunk.append(best_idx)
                remaining.remove(best_idx)
                chunk_entities.add(self.e1_ids[best_idx])
                chunk_entities.add(self.e2_ids[best_idx])

            chunks.append(chunk)

        return chunks

    def _make_batches(self):
        keys = list(self.doc_keys)
        if self.shuffle_docs:
            random.shuffle(keys)
        else:
            keys = sorted(keys, key=lambda d: len(self.doc_groups[d]), reverse=True)

        batches = []
        for key in keys:
            doc_indices = self.doc_groups[key]
            doc_chunks = self._make_doc_chunks(doc_indices)
            batches.extend(doc_chunks)

        return batches

    def __iter__(self):
        if self.shuffle_docs:
            self.batches = self._make_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class DocBatchSampler(Sampler):
    """
    BatchSampler that ensures all samples from the same document 
    are in the same batch unless max_pairs_per_batch is set and a single
    document exceeds that budget (then it is split into chunks).
    """
    def __init__(self, doc_ids, batch_size, shuffle=True, max_pairs_per_batch=None):
        self.doc_ids = doc_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_pairs_per_batch = max_pairs_per_batch if (max_pairs_per_batch is not None and max_pairs_per_batch > 0) else None
        
        # Group indices by doc_id
        self.doc_groups = {}
        for idx, doc_id in enumerate(self.doc_ids):
            d = str(doc_id)
            if d not in self.doc_groups:
                self.doc_groups[d] = []
            self.doc_groups[d].append(idx)
        for d in self.doc_groups:
            self.doc_groups[d].sort()
            
        self.doc_keys = sorted(self.doc_groups.keys(), key=_doc_sort_key)
        # Pre-calculate to have exact __len__
        self.batches = self._make_batches()

    def _build_doc_units(self, keys):
        units = []
        for key in keys:
            indices = self.doc_groups[key]
            if self.max_pairs_per_batch is None or len(indices) <= self.max_pairs_per_batch:
                units.append({'doc_key': key, 'indices': indices, 'pairs': len(indices)})
                continue

            chunk_size = self.max_pairs_per_batch
            for start in range(0, len(indices), chunk_size):
                chunk = indices[start:start + chunk_size]
                units.append({'doc_key': key, 'indices': chunk, 'pairs': len(chunk)})
        return units
        
    def _make_batches(self):
        keys = list(self.doc_keys)
        if self.shuffle:
            random.shuffle(keys)

        # Build document units (possibly split for oversized docs).
        units = self._build_doc_units(keys)

        # Deterministic mode: sort heavy units first then balance.
        # Shuffle mode: randomize unit order.
        if self.shuffle:
            random.shuffle(units)
        else:
            units = sorted(units, key=lambda u: (-u['pairs'], _doc_sort_key(u['doc_key'])))

        bins = []
        for unit in units:
            doc_pairs = unit['pairs']

            candidates = []
            for i, b in enumerate(bins):
                if len(b['docs']) >= self.batch_size:
                    continue
                if self.max_pairs_per_batch is not None:
                    if b['pairs'] + doc_pairs > self.max_pairs_per_batch and len(b['docs']) > 0:
                        continue
                candidates.append(i)

            if candidates:
                best_idx = min(candidates, key=lambda i: (bins[i]['pairs'], len(bins[i]['docs'])))
                bins[best_idx]['docs'].append(unit)
                bins[best_idx]['pairs'] += doc_pairs
            else:
                bins.append({'docs': [unit], 'pairs': doc_pairs})

        batches = []
        for b in bins:
            batch_indices = []
            for unit in sorted(b['docs'], key=lambda u: (_doc_sort_key(u['doc_key']), -u['pairs'])):
                batch_indices.extend(unit['indices'])
            batches.append(batch_indices)

        return batches

    def __iter__(self):
        # Re-generate if shuffle is on
        if self.shuffle:
            self.batches = self._make_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
