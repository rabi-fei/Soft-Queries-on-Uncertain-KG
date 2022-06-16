# Architecture Design for Truth Value Knowledge Graph Reasoning

The functionality of this framework handles 3 data objects in 3 parts

- Data objects
    - [D0] Index for entities and relations
    - [D1] KG, knowledge graph, triples
    - [D2] NBP, neural binary predicate, embedding and neural network models
    - [D3] Sentence / Query
- Pipelines
    - [P1] Training
        - Input: [D0] and [D1]
        - Output: [D2]
    - [P2] Inference
        - Input: [D0], [D3] and [D1] or [D2]
        - Output: predict booleans/ entity index from [D0]
    - [P3] Task Sampling
        - Input: [D0], [D1]-full version, [D1]-observed version, [D3]
        - Output: [D4]

## 1 Data object interfaces

- **[D0] Index for entities and relations**
    Beyond this level, we handle the symbolic ids. All data placed on MEM
    - Properties
        - string to id
        - id to string
    - Methods
        - from raw string to id
        - from id to raw string
- **[D1] KnowledgeGraph**
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
- **[D2] NBP**
    neural network predicates
    - methods
        - embedding_score(head_emb, rel_emb, tail_emb)
        - batch_predicate_score(triple_tensor)
        - score2prob
- **[D3] First order formula type**
- **[D4] First order formula sample**

## 2 Process pipeline

- **[P1] Training**
    1. The overall pipeline
        1. compute the minibatch objective
            - Iterator: **method of KG**
                - [ ] TripleIterator
                    - **params**: `batch_size`
                    - **input**: NA
                    - **output**: batch of triples
                - [ ] NodeIterator
                    - **params**: `batch_size`
                    - **input**: NA
                    - **output**: batch of nodes
            - Sampler (node to triple)
                - [ ] `TripleSampler` not necessary when using the triple
                    - **input**: batch of triples
                    - **output**: batch of triple lists with size 1
                - [ ] `SubgraphSampler` k-hop subgraph, then subgraph triples
                    - **params**: `num_hops`
                    - **input**: batch of nodes
                    - **output**: batch of subgraph triple list
                - [ ] `EFSampler` k-round EF game, then subgraph triples
                    - **params**: `num_rounds`, `beam_size`, `margin` (in case one should transform the scores to probabilities)
                    - **input**: batch of nodes
                    - **output**: batch of subgraph triple list
            - (Subgraph) Negative Sampler: extend the triple to subgraph, each triple is a 2 subgraph
                - [ ] negative sample of each triple
                    - **params**: `num_negative_samples`
                    - **input**: batch of triple list `[batch_size, size_of_triple_list]`
                    - **output**: batch of triple list `[batch_size, size_of_triple_list, num_negative_samples]`
                - [ ] negative sample of each subgraph
                    - **params**: `num_negative_samples`
                    - **input**: batch of triple list `[batch_size, size_of_triple_list]`
                    - **output**: batch of triple list `[batch_size, size_of_triple_list, num_negative_samples]`
            - Objective Function: what we use to compute after the positive and negative graphs
                - [ ] Ranking loss
                    - params `margin`
                - [ ] NCE loss
                    - params `k_noise`
        2. after the minibatch objective is compute
            - optimization
            - evaluation
            - logging
            - checkpoint saving

- **[P2] Inference**
    - Formula
        - data structure level
            - placeholders for anchor nodes, relations
            - existential variables
            - free variables
        - Serialization to string
        - Instantiation from batch of input data samples, i.e., fill up the placeholder.
    - Formula Sample, instantiated from the formula type
    - Reasoning Machine
        - Infer
            - Given the formula sample (of the same type) and the model, (KG or NBP), return the answer
        - Learn
            - Given the formula sample (of the sampe type), NBP, and the observed answer set, update the parameters of the trainable model


## 3. Formula Details

The samples are formulated by the formulas and the answers convension
- when all vars are quantified, the answer should be in {0, 1}
- when k vars are free, the answer should be set of k-triples {(a_1, ..., a_k)}
- formula are described by the following three row strings

```text
FVars:V1,...,Vk // claim free variables
Evars:W1,...,Wk // claim existential variables
Uvars:U1,...,Uk // claim univerrsal variables
Formula:String // defined by a Context Free Grammar
```

The context free grammar reads
```text

Formula   := (Formula)
             !(Formula)
             (Formula)&(Formula)
             (Formula)|(Formula)
             Atom

Atom      := [Relation](Term,Term)

Relation  := Int, indicates the relation id

Term      := Entity
             Variable

Entity    := Int, indicates the entity id

Variable  := Str, given in the EVars, UVars, and FVars

```

## Code Design



## Nvidia Running Log
```
ngc batch run --name "ml-model.TransE_Iso_FB15k.baseline" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "sh /mount/efg/run_batch.sh config/TransE-FB15K.yaml" --result /results --image "nvidia/pytorch:22.03-py3" --org nvidian --team sae --datasetid 97639:/mount/ckgs --workspace 5xgjEDXoR7GwHTw559lEBQ:/mount/efg:RW
```

| name                              | Model  | Config File                      | Comand                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------- | ------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ml-model.TransE.Iso_FB15K_Classic | TransE | config/TransE-FB15K-classlc.yaml | `ngc batch run --name "ml-model.TransE.Iso_FB15K_Classic" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "sh /mount/efg/run_batch.sh config/TransE-FB15K-Classic.yaml" --result /results --image "nvidia/pytorch:22.03-py3" --org nvidian --team sae --datasetid 97639:/mount/ckgs --workspace 5xgjEDXoR7GwHTw559lEBQ:/mount/efg:RW` |
| ml-model.TransE.Iso_FB15K_Modern  | TransE | config/TransE-FB15K-classic.yaml | `ngc batch run --name "ml-model.TransE.Iso_FB15K_Modern" --preempt RUNONCE --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "sh /mount/efg/run_batch.sh config/TransE-FB15K-Modern.yaml" --result /results --image "nvidia/pytorch:22.03-py3" --org nvidian --team sae --datasetid 97639:/mount/ckgs --workspace 5xgjEDXoR7GwHTw559lEBQ:/mount/efg:RW` |
|                                   |        |                                  |                                                                                                                                                                                                                                                                                                                                                                    |
