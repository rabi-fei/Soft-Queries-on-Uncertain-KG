from src.language import fol


test_formula = """(disj,
                    (conj,
                        (neg,(pred,P,(A),(B))),
                        (pred,Q,(C),(A))),
                    (conj,
                        (pred,P,(C),(B)),
                        (neg,(pred,Q,(A),(B)))
                    )"""

fvar1 = fol.get_ldict(fol.Term.op,
                      name='fvar1', state=fol.Term.FREE, entity_id_list=[])

fvar2 = fol.get_ldict(fol.Term.op,
                      name='fvar2', state=fol.Term.FREE, entity_id_list=[])

evar1 = fol.get_ldict(fol.Term.op,
                      name='evar1', state=fol.Term.EXISTENTIAL, entity_id_list=[])

uvar1 = fol.get_ldict(fol.Term.op,
                      name='uvar1', state=fol.Term.UNIVERSAL, entity_id_list=[])
lit1  = fol.get_ldict(fol.Term.op,
                      name='lit1', state=fol.Term.LITERAL, entity_id_list=[])

lit2 = fol.get_ldict(fol.Term.op,
                     name='lit2', state=fol.Term.LITERAL, entity_id_list=[])

atom1 = fol.get_ldict(fol.BinaryPredicate.op,
                      name='f1f2',
                      relation_id_list=[],
                      term1=fvar1,
                      term2=fvar2)

natom1 = fol.get_ldict(fol.Negation.op,
                       name='na1',
                       formula=atom1)

atom2 = fol.get_ldict(fol.BinaryPredicate.op,
                      name='f1e1',
                      relation_id_list=[],
                      term1=fvar1,
                      term2=evar1)

atom3 = fol.get_ldict(fol.BinaryPredicate.op,
                      name='f2u1',
                      relation_id_list=[],
                      term1=fvar2,
                      term2=uvar1)

atom4 = fol.get_ldict(fol.BinaryPredicate.op,
                      name='f2l1',
                      relation_id_list=[],
                      term1=fvar2,
                      term2=lit1)

atom5 = fol.get_ldict(fol.BinaryPredicate.op,
                      name='e1l2',
                      relation_id_list=[],
                      term1=evar1,
                      term2=lit2)


clause1 = fol.get_ldict(fol.Disjunction.op,
                        formulas=[natom1, atom2, atom3])

clause2 = fol.get_ldict(fol.Disjunction.op,
                        formulas=[atom4, atom5])

ldict = fol.get_ldict(fol.Conjunction.op, formulas=[clause1, clause2])


if __name__ == "__main__":
    print(ldict)
    print(fol.Formula.parse(ldict))