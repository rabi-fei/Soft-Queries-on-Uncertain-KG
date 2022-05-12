from language import fof


test_formula = """(disj,
                    (conj,
                        (neg,(pred,P,(A),(B))),
                        (pred,Q,(C),(A))),
                    (conj,
                        (pred,P,(C),(B)),
                        (neg,(pred,Q,(A),(B)))
                    )"""

fvar1 = fof.get_ldict(fof.Term.op,
                      name='fvar1', state=fof.Term.FREE, entity_id_list=[])

fvar2 = fof.get_ldict(fof.Term.op,
                      name='fvar2', state=fof.Term.FREE, entity_id_list=[])

evar1 = fof.get_ldict(fof.Term.op,
                      name='evar1', state=fof.Term.EXISTENTIAL, entity_id_list=[])

uvar1 = fof.get_ldict(fof.Term.op,
                      name='uvar1', state=fof.Term.UNIVERSAL, entity_id_list=[])
lit1  = fof.get_ldict(fof.Term.op,
                      name='lit1', state=fof.Term.SYMBOL, entity_id_list=[])

lit2 = fof.get_ldict(fof.Term.op,
                     name='lit2', state=fof.Term.SYMBOL, entity_id_list=[])

atom1 = fof.get_ldict(fof.BinaryPredicate.op,
                      name='f1f2',
                      relation_id_list=[],
                      term1=fvar1,
                      term2=fvar2)

natom1 = fof.get_ldict(fof.Negation.op,
                       name='na1',
                       formula=atom1)

atom2 = fof.get_ldict(fof.BinaryPredicate.op,
                      name='f1e1',
                      relation_id_list=[],
                      term1=fvar1,
                      term2=evar1)

atom3 = fof.get_ldict(fof.BinaryPredicate.op,
                      name='f2u1',
                      relation_id_list=[],
                      term1=fvar2,
                      term2=uvar1)

atom4 = fof.get_ldict(fof.BinaryPredicate.op,
                      name='f2l1',
                      relation_id_list=[],
                      term1=fvar2,
                      term2=lit1)

atom5 = fof.get_ldict(fof.BinaryPredicate.op,
                      name='e1l2',
                      relation_id_list=[],
                      term1=evar1,
                      term2=lit2)


clause1 = fof.get_ldict(fof.Disjunction.op,
                        formulas=[natom1, atom2, atom3])

clause2 = fof.get_ldict(fof.Disjunction.op,
                        formulas=[atom4, atom5])

ldict = fof.get_ldict(fof.Conjunction.op, formulas=[clause1, clause2])


if __name__ == "__main__":
    print(ldict)
    lobject = fof.Formula.parse(ldict)
    print(lobject)
    print(lobject.lstr())