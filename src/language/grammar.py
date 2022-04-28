"""
Parse the grammar into the classes
Formula = (Formula)
        = Formula|Formula
        = Formula&Formula
        = !Formula
        = Predicate(Term,Term)
Predicate = r[number]
Term = e[number]
     = u[number]
     = l[number]
     = f[number]
"""

from re import A
from .fol import Conjunction, Disjunction, Lobject, Negation, BinaryPredicate, Term

def remove_outmost_backets(lstr: str):
    if not (lstr[0] == '(' and lstr[-1] == ')'):
        return lstr

    bracket_stack = []
    for i, c in enumerate(lstr):
        if c == '(':
            bracket_stack.append(i)
        elif c == ')':
            left_bracket_index = bracket_stack.pop(-1)

    assert len(bracket_stack) == 0
    if left_bracket_index == 0:
        return lstr[1:-1]
    else:
        return lstr

def remove_brackets(lstr: str):
    new_lstr = remove_outmost_backets(lstr)
    while new_lstr != lstr:
        lstr = new_lstr
        new_lstr = remove_outmost_backets(lstr)
    return lstr



def map_term_name_to_type(name: str):
    c = name[0]
    if c == 'e':
        return Term.EXISTENTIAL, True
    elif c == 'f':
        return Term.FREE, True
    elif c == 'u':
        return Term.UNIVERSAL, True
    elif c == 'l':
        return Term.LITERAL, True
    else:
        assert name.isnumeric()
        term_id = int(name)
        return term_id, False

def parse_term(term_name):
    term_state, is_abstract = map_term_name_to_type(term_name)
    if is_abstract:
        term = Term(state=term_state, name=term_name)
    else:
        term = Term(state=Term.LITERAL,
                    name="literal_by_id",
                    entity_id_list=[term_state])
    return term


def parse_lstr(lstr: str) -> Lobject:
    """
    parse the string a.k.a, lformula to lobject
    """
    _lstr = remove_brackets(lstr)

    # identify top-level operator
    if lstr[0] == '!':
        sub_lstr = _lstr[1:]
        sub_formula = parse_lstr(sub_lstr)
        return Negation(formula=sub_formula)

    binary_operator_index = -1
    binary_operator = ""
    for i, c in enumerate(_lstr):
        if c in "&|":
            binary_operator_index = i
            binary_operator = c

    if binary_operator_index >= 0:
        left_lstr = _lstr[:binary_operator_index]
        left_formula = parse_lstr(left_lstr)
        right_lstr = _lstr[binary_operator_index+1:]
        right_formula = parse_lstr(right_lstr)
        if binary_operator == '&':
            return Conjunction(formulas=[left_formula, right_formula])
        if binary_operator == '|':
            return Disjunction(formulas=[left_formula, right_formula])

    else:  # parse predicate
        assert _lstr[-1] == ')'
        predicate_name, right_lstr = _lstr.split('(')
        right_lstr = right_lstr[:-1]
        term1_name, term2_name = right_lstr.split(',')

        term1 = parse_term(term1_name)
        term2 = parse_term(term2_name)
        if predicate_name.isnumeric():
            predicate_id = int(predicate_name)
            predicate = BinaryPredicate(name="predicate_by_id",
                                        relation_id_list=[predicate_id],
                                        term1=term1,
                                        term2=term2)
        else:
            predicate = BinaryPredicate(name=predicate_name,
                                        relation_id_list=[],
                                        term1=term1,
                                        term2=term2)

        return predicate
