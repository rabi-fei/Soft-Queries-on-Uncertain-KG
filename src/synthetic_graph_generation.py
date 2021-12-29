import argparse
import random
import os
from collections import Counter, defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--graph_name", default='family')


def split_graph_with_full_entities(triples, loss_ratio):
    """
    triples: graph triples
    loss_ratio: some part of the triples should be splitted
    """
    node2id = {}
    rel2id = {}

    # check the index is already done, if the input is some sort of integers
    already_indexed = False
    # first check the h, r, t are integer
    if isinstance(triples[0][0], int) and isinstance(triples[0][1], int):
        already_indexed = True

    # then check the entity are relation are set(range(size))
    if already_indexed is True:
        entity_set = set()
        relation_set = set()
        for h, r, t in triples:
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)
        if entity_set != set(range(len(entity_set))):
            already_indexed = False
        if relation_set != set(range(len(relation_set))):
            already_indexed = False

    print("already indexed", already_indexed)

    if already_indexed:
        node2id = {i: i for i in range(len(entity_set))}
        rel2id = {i: i for i in range(len(relation_set))}
    else:
        print("reindexing")
        for h, r, t in triples:
            if h not in node2id:
                node2id[len(node2id)] = h
            if t not in node2id:
                node2id[len(node2id)] = t
            if r not in rel2id:
                rel2id[len(rel2id)] = r

    pick_triples = {}
    loss_triples = {}
    index = {}
    node_counter = Counter()
    h2triple = defaultdict(list)
    t2triple = defaultdict(list)
    for i, (h, r, t) in enumerate(triples):
        pick_triples[i] = (node2id[h], rel2id[r], node2id[t])
        index[i] = (node2id[h], rel2id[r], node2id[t])
        node_counter[node2id[h]] += 1
        node_counter[node2id[t]] += 1
        h2triple[h].append(i)
        t2triple[t].append(i)

    def loss(i):
        assert i in pick_triples
        assert i not in loss_triples
        loss_triples[i] = index[i]
        del pick_triples[i]

    def pick(i):
        assert i not in pick_triples
        assert i in loss_triples
        pick_triples[i] = index[i]
        del loss_triples[i]

    def check_integrety():
        pick_node_counter = Counter()
        for h, r, t in pick_triples.values():
            pick_node_counter[h] += 1
            pick_node_counter[t] += 1

        if len(pick_node_counter) == len(node_counter):
            print('all entity preserved, happy ending')
            return None
        else:
            print('not all entity preserved')
            return (pick_node_counter,
                    set(node_counter.keys()).difference(set(pick_node_counter.keys())))

    # first we sample
    for i in range(len(index)):
        if random.random() < loss_ratio:
            loss(i)

    # then check the integrety
    check_res = check_integrety()
    while check_res:
        pick_node_counter, left_node_set = check_res
        left_nodes = list(left_node_set)
        for l_node in left_nodes:
            possible_triples_to_pick = h2triple[l_node] + t2triple[l_node]
            for tri in possible_triples_to_pick:
                if tri not in pick_triples:
                    pick(tri)
                    break
            p_node = pick_node_counter.most_common(1)[0]
            possible_triples_to_loss = h2triple[p_node] + t2triple[p_node]
            for tri in possible_triples_to_loss:
                if tri not in loss_triples:
                    loss(tri)
                    break
        check_res = check_integrety()

    return list(pick_triples.values()), list(loss_triples.values())


class People:
    STATE_SINGLE = 1
    STATE_MERRAGE = 2
    STATE_DEATH = 3

    GENDER_MALE = 4
    GENDER_FEMALE = 5

    def __init__(self, gender, pid):
        self.gender = gender
        self.pid = pid
        self.status = self.STATE_SINGLE

    @classmethod
    def born(cls, pid):
        gender = random.choice([cls.GENDER_MALE, cls.GENDER_FEMALE])
        return cls(gender=gender, pid=pid)

    def die(self):
        self.status = self.STATE_DEATH

    def marry(self):
        self.status = self.STATE_MERRAGE

    def is_single(self):
        return self.status == self.STATE_SINGLE

    def is_male(self):
        return self.gender == self.GENDER_MALE


class FamilySimulator:

    REL_ISFATHEROF = 0
    REL_ISMOTHEROF = 1
    REL_MARRYWITH = 2
    REL_ISFATHERINLAW = 3
    REL_ISMOTHERINLAW = 4
    REL_INIT_PEOPLE = 5

    def __init__(self, num_gen, num_init_people, birth_ratio, death_ratio):
        self.num_gen = num_gen
        self.num_init_people = num_init_people
        self.birth_ratio = birth_ratio
        self.death_ratio = death_ratio
        self.pid2people = {}
        self.triples = {}
        self.stats = Counter()

    def dump_tsv(self, loss_ratio, target_folder, requires_dev=True):
        os.makedirs(target_folder, exist_ok=True)

        train_triples, test_triples = split_graph_with_full_entities(
            self.triples, loss_ratio)
        train_triples, dev_triples = split_graph_with_full_entities(
            train_triples, loss_ratio)

        def dump_triples(triples, target_file):
            lines = []
            for h, r, t in triples:
                lines.append(f'{h}\t{r}\t{t}\n')

            with open(target_file, 'wt') as f:
                f.writelines(lines)

        train_file = os.path.join(target_folder, 'train.tsv')
        dump_triples(train_triples, train_file)
        dev_file = os.path.join(target_folder, 'dev.tsv')
        dump_triples(dev_triples, dev_file)
        test_file = os.path.join(target_folder, 'test.tsv')
        dump_triples(test_triples, test_file)

    def simulate_next_generation(self):
        single_males = []
        single_females = []

        generation_counter = Counter()

        for pid in self.pid2people:
            p = self.pid2people[pid]
            if random.random() < self.death_ratio:
                p.die()
                generation_counter['death'] += 1
                self.stats['death'] += 1
                continue
            if p.is_single():
                if p.is_male():
                    single_males.append(p)
                else:
                    single_females.append(p)

        random.shuffle(single_females)
        random.shuffle(single_males)

        for m, f in zip(single_males, single_females):
            b = self.marry(m, f)
            generation_counter['birth'] += b
            self.stats['birth'] += b

        print("Generation Counter", generation_counter)
        return

    def marry(self, m, f):
        # register
        self.triples.append(
            (m.pid, self.REL_MARRYWITH, f.pid)
        )
        self.triples.append(
            (f.pid, self.REL_MARRYWITH, m.pid)
        )

        m.marry()
        f.marry()

        if m.pid in self.father_lookup and m.pid in self.mother_lookup:
            father = self.father_lookup[m.pid]
            mother = self.mother_lookup[m.pid]

            self.triples.append(
                (father, self.REL_ISFATHERINLAW, f.pid)
            )
            self.triples.append(
                (mother, self.REL_ISMOTHERINLAW, f.pid)
            )

        if f.pid in self.father_lookup and f.pid in self.mother_lookup:
            father = self.father_lookup[f.pid]
            mother = self.mother_lookup[f.pid]

            self.triples.append(
                (father, self.REL_ISFATHERINLAW, m.pid)
            )
            self.triples.append(
                (mother, self.REL_ISMOTHERINLAW, m.pid)
            )

        # born
        born_counter = 0
        while True:
            if born_counter > 0 and random.random() < 1 - self.birth_ratio / 2:
                break
            new_p = People.born(len(self.pid2people))
            born_counter += 1
            self.triples.append(
                (m.pid, self.REL_ISFATHEROF, new_p.pid)
            )
            self.triples.append(
                (f.pid, self.REL_ISMOTHEROF, new_p.pid)
            )

            self.father_lookup[new_p.pid] = m.pid
            self.mother_lookup[new_p.pid] = f.pid

            self.pid2people[new_p.pid] = new_p

        return born_counter

    def generate(self):
        self.pid2people = {}
        self.father_lookup = {}
        self.mother_lookup = {}
        self.triples = []
        self.stats = Counter()

        for pid in range(self.num_init_people):
            self.pid2people[pid] = People.born(pid)
            self.stats['birth'] += 1
            self.triples.append((pid, self.REL_INIT_PEOPLE, pid))

        for _ in range(self.num_gen):
            self.simulate_next_generation()
            print("global stats", self.stats)
            print("people alive", len(self.pid2people) +
                  self.stats['birth'] - self.stats['death'])

        print(len(self.pid2people))
        print(len(self.triples))
        relation_types = [r for _, r, _ in self.triples]
        relation_counter = Counter(relation_types)
        print(relation_counter)


def family_kg_generate(
        num_gen=5,
        num_init_people=1000,
        birth_ratio=1.2,
        death_ratio=0.2,
        loss_ratios=[0.05, 0.1, 0.2, 0.4],
        target_path='data',
        **kwargs):
    simulator = FamilySimulator(num_gen=num_gen,
                                num_init_people=num_init_people,
                                birth_ratio=birth_ratio,
                                death_ratio=death_ratio)
    simulator.generate()
    os.makedirs(target_path, exist_ok=True)
    for lr in loss_ratios:
        simulator.dump_tsv(
            loss_ratio=lr, target_folder=os.path.join(target_path, f"family-loss-{lr}"))


def kg_generate(args):

    if args.graph_name == 'family':
        family_kg_generate()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    kg_generate(args)
