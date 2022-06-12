from src.pipeline.reasoning_machine import GradientReasoningMachineEFO
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import TransE
from src.utils.data import QueryAnsweringSeqDataLoader, TrainQueryAnsweringWithSentenceVerificationDataLoader

if __name__ == "__main__":
    kgidx = KGIndex.load("data/FB15k-237-betae/kgindex.json")
    kg = KnowledgeGraph.create("data/FB15k-237-betae/train_kg.tsv", kgidx)
    nbp = TransE(kgidx.num_entities, kgidx.num_relations, 100, 1, 1, 'cpu')
    train_dataloader = TrainQueryAnsweringWithSentenceVerificationDataLoader(
        'data/FB15k-237-betae/train-qaa.json',
        answer_size=kgidx.num_entities,
        neg_sample_size=128,
        batch_size=7,
        shuffle=True,
        num_workers=0)
    test_valid_dataloader = QueryAnsweringSeqDataLoader(
        'data/FB15k-237-betae/test-qaa.json',
        batch_size=7,
        shuffle=False,
        num_workers=0
    )
    grm = GradientReasoningMachineEFO(
        reasoning_rate=1e-1,
        reasoning_steps=20,
        reasoning_optimizer='Adam',
        nbp=nbp)
    for i, fofs in enumerate(train_dataloader):
        grm.reasoning(fofs)
        if i > 100: break

    for i, fofs in enumerate(test_valid_dataloader):
        grm.reasoning(fofs)
        if i > 100: break
