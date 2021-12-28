# Ehrenfeucht–Fraı̈sśe Learning for Knowledge Graph Embedding and Reasoning


## Design doc

### 1. Model
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
```
FVars:V1,...,Vk // claim free variables
Evars:W1,...,Wk // claim existentially quantified variables
Formula:String by a Context Free Grammar
```
The context free grammar for the formula
```
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