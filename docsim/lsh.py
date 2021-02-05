from collections import defaultdict
import numpy as np

from docsim import utils


class MinHashLSH:
    def __init__(self, documents: list, signatures: np.array, num_bands: int):
        self.documents = documents
        self.signature_matrix = signatures
        self.num_documents = self.signature_matrix.shape[1]
        self.num_bands = num_bands
        self.band_hash_tables = [defaultdict(list) for _ in range(num_bands)]
        self.doc_candidates = {i: set() for i in range(self.num_documents)}

    def build(self):
        """
        Perform locality sensitive hashing on the provided min hash signature matrix
        """
        bands = np.split(self.signature_matrix, self.num_bands)
        for i, band in enumerate(bands):
            # The (partial) signatures within a band
            banded_signatures = np.hsplit(band, self.num_documents)
            for doc_idx, banded_signature in enumerate(banded_signatures):
                # Convert to a hashable type
                banded_signature = tuple(banded_signature.flatten().astype(int))
                self.band_hash_tables[i][banded_signature].append(doc_idx)

        # Iterate over the buckets to find all candidates
        self._extract_candidates()
        self._filter_self_candidates()

    def query(self, doc_idx: int, threshold: float) -> list:
        """
        Given a document and a threshold, finds other documents in the corpus that have similarity above the threshold.
        :param doc_idx: document index
        :param threshold: similarity threshold
        :return: list of indexes of similar documents
        """
        query_doc = self.documents[doc_idx]
        candidates = self.doc_candidates[doc_idx]
        similar_docs = []
        for candidate_idx in candidates:
            candidate_doc = self.documents[candidate_idx]
            similarity = utils.jaccard(set(query_doc), set(candidate_doc))
            if similarity >= threshold:
                similar_docs.append(candidate_idx)
        return similar_docs

    def _extract_candidates(self):
        for bucket in self.band_hash_tables:
            doc_ids = list(filter(lambda x: len(x) > 1, list(bucket.values())))
            for candidates in doc_ids:
                for doc_id in candidates:
                    self.doc_candidates[doc_id].update(candidates)

    def _filter_self_candidates(self):
        for doc_idx in self.doc_candidates:
            if doc_idx in self.doc_candidates[doc_idx]:
                self.doc_candidates[doc_idx].remove(doc_idx)
