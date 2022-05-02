from src.pipeline.reasoning_machine import GradientReasoningMachine
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import TransE
from src.utils.data import QueryAnsweringSeqDataLoader

if __name__ == "__main__":
    kgidx = KGIndex.load("data/FB15k-237-betae/kgindex.json")
    kg = KnowledgeGraph.create("data/FB15k-237-betae/train_kg.tsv", kgidx)
    nbp = TransE(kgidx.num_entities, kgidx.num_relations, 100, 1, 'cpu')
    qad = QueryAnsweringSeqDataLoader('data/FB15k-237-betae/test-qaa.json',
                                      batch_size=7,
                                      shuffle=True,
                                      num_workers=2)
    grm = GradientReasoningMachine(
        reasoning_rate=1e-4,
        reasoning_steps=100,
        reasoning_optimizer='Adam',
        nbp=nbp)
    for fofs in qad:
        grm.reasoning(fofs)
