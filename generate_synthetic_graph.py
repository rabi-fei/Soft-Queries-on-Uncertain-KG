import argparse
import random
import os
from collections import Counter, defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--graph_name", default='family')

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
        self.triples = []
        self.stats = Counter()

    # TODO: dump with index
    def dump_tsv(self, target_folder, num_folds=5):
        os.makedirs(target_folder, exist_ok=True)

        def dump_triples(triples, target_file):
            lines = []
            for h, r, t in triples:
                lines.append(f'{h}\t{r}\t{t}\n')

            with open(target_file, 'wt') as f:
                f.writelines(lines)

        rel_triple_splits = {}
        for h, r, t in self.triples:
            pass

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
