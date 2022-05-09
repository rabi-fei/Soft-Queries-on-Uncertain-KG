from src.pipeline.reasoning_machine import GradientReasoningMachine
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import TransE
from src.utils.data import QueryAnsweringSeqDataLoader

if __name__ == "__main__":
    kgidx = KGIndex.load("data/FB15k-237-betae/kgindex.json")
    kg = KnowledgeGraph.create("data/FB15k-237-betae/train_kg.tsv", kgidx)
    nbp = TransE(kgidx.num_entities, kgidx.num_relations, 100, 1, 1, 'cpu')
    qad = QueryAnsweringSeqDataLoader('data/FB15k-237-betae/train-qaa.json',
                                      batch_size=7,
                                      shuffle=True,
                                      num_workers=0)
    grm = GradientReasoningMachine(
        reasoning_rate=1e-1,
        reasoning_steps=20,
        reasoning_optimizer='Adam',
        nbp=nbp)
    for fofs in qad:
        grm.inference(fofs)
