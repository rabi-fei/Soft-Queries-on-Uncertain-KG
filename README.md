# Ehrenfeucht–Fraı̈sśe Learning for Knowledge Graph Embedding and Reasoning

An framework for Knowlege Graph Embedding Learning and Verification

## Design doc

The functionality of this framework handles 3 data objects in 3 parts

- Data objects
  - [D0] Index for entities and relations
  - [D1] KG, knowledge graph, triples
  - [D2] NBP, neural binary predicate, embedding and neural network models
  - [D3] Sentence / Query

- Processes
  - Training
    - Input: [D0] and [D1]
    - Output: [D2]
  - Inference
    - Input: [D0], [D3] and [D1] or [D2]
    - Output: predict booleans/ entity index from [D0]
  - Task Sampling
    - Input: [D0], [D1]-full version, [D1]-observed version, [D3]
    - Output: [D4]

### Data object interfaces

#### [D0] Index for entities and relations

Beyond this level, we handle the symbolic ids. All data placed on MEM

- Properties
  - string to id
  - id to string
- Methods
  - from raw string to id
  - from id to raw string


#### [D1] KnowledgeGraph

Try to stored in CUDA for parallel lookup

- Properties
  - triples: a list of (h, r, t) triples, total triple N
  - triple_tensor: a tensor of shape (N, 3)
  - triple_index, sparse tensor T[h, r, t] = 1 iff (h, r, t) in triples
  - dconnect_tensor, a tensor of shape (N, 2) for all directed edges
  - dconnect_index, sparse tensor T[h, t] = 1 iff (h, t) is connected directly
- Methods
  - create
  - from_config
  - get_triple_dataloader
  - get_sub_graph(self, entities)
  - get_non_neighbor_triple(self, entities, k, reverse)

#### [D2] NBP

neural network prediction

- methods
  - embedding_score(head_emb, rel_emb, tail_emb)
  - batch_predicate_score(triple_tensor)

#### [D3] Sentence Type / Query Type

#### [D4] Sentence Sample / Query Sample


### 1. Training

Model discussed here is relational, that is, we only consider the predicate.

Two kinds of models are implemented, the one is `KnowledgeGraph` and the other
one is `NeuralBinaryPredicate`

- Utility
  - `load_triple_from_tsv(triple_files)`
    - desc: return an iterator for triples
  - `RaggedBatch(flattened_batch, sample_sizes)`
    - desc: batch data with uncertain sizes for each sample
  - `evaluate(target_kg, observed_kg, nbp)`
    - desc: evaluate the target_kg on nbp given the observed_kg under the filter setting
- Common Interface
  - create
- Specific Interface
  - `KnowledgeGraph`
    - `get_connected_entities(entity_batch_tensor)`
    - `get_connected_relations(entity_batch_tensor)`
    - `get_head(entity_batch_tensor, relation_batch_tensor)`
    - `get_tail(entity_batch_tensor, relation_batch_tensor)`
    - `get_rel(head_entity_batch_tensor, tail_entity_batch_tensor)`
  - `NeuralBinaryPredicate`
    - `batch_triple_score(head_batch_tensor, rel_batch_tensor, tail_batch_tensor)`

### 2. Learner

- `IsomorphicLearner`
- `ElementaryLearner`

### 3. Formula

The samples are formulated by the formulas and the answers
convension

- only considers existential quantifier
- when all vars are quantified, the answer should be in {0, 1}
- when k vars are free, the answer should be set of k-triples {(a_1, ..., a_k)}
- formula are described by the following three row strings

```text
FVars:V1,...,Vk // claim free variables
Evars:W1,...,Wk // claim existentially quantified variables
Uvars:U1,...,Uk // claim existentially quantified variables
Formula:String by a Context Free Grammar
```

exists w1, for all u1, p(w1, a) and p(u1, w1) and p(w1, u1)

max w1
  min u1
    p(w1, a) and p(u1, w1) and p(w1, u1)

The context free grammar for the formula

```text
Formula   := Atom
             !(Atom)
             (Atom)&(Atom)
             (Atom)|(Atom)
Atom      := [Relation](Term,Term)
Relation  := Int, indicates the relation id
Term      := Ent
             Var
Entity    := Int, indicates the entity id
Variable  := Str, given in the FVars or Evars
```
