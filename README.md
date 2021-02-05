# document-similarity

A minimal implementation of document similarity using MinHash LSH.


# Quick start

```bash
pip install -r requirements.txt
```

## Building signatures

```python
sample_sentences = ["the fox jumped over the democracy", 
                    "there's always money in the banana stand",
                    "the revolution will be live streamed by the fox"]
# Shingle the documents
shingled_sentences = [docsim.utils.generate_ngrams(sentence.split(), 2) for sentence in sample_sentences]

# Build minhash signatures of the shingled documents
minhash = MinHash(num_hashes=5)
signatures = [minhash.signature(shingled_sentence) for shingled_sentence in shingled_sentences]
```

## Building LSH index
```python
# Convert list of signatures into a 2d array
sig_matrix = np.array(signatures).T

# Build LSH index
minhash_lsh = MinHashLSH(documents=sample_sentences, 
                         signatures=sig_matrix, 
                         num_bands=2)
minhash_lsh.build()

# Get all candidates
doc_candidates = minhash_lsh.doc_candidates
```


Refer to the Jupyter notebook for a full demonstration on usage.




