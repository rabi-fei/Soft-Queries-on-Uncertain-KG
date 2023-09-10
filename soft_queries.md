# On soft reasoning over uncertain knowledge graph

## Query formula with alpha and beta parameter

r1(s1,e1,50%,1.0), where 50% are percentile of relation's confidence and 1.0 are weight assigned for constraint's importance.

For necessity requirement, we select three level, zero(0%), low(25%), and normal(50%). In postive requirement, the candidates not meeting the requirements will be rejected. In zero requirment, under the plus operation, the answers don't need to meet all the constraints.


## Sampling strategy for meaning soft queries.

In postive requirement, the strategy is similar as classical logical queries, the answers must meet all the constraint and the requirements is more strict.

In zero requirement, part satified constraints also results in answers. We select the shortest path that originates from the anchor and terminates at the free node, ensuring the presence of a valid reasoning chain. The ohter grounded in query graph is valid for tail relation constraint. The other grounding of the query graph is valid, considering the tail relation constraint.

For negative edges in the query graph, we begin by grounding the positive graph and then choose the negative edges which will change the final answers.