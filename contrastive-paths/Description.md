# Contrastive Path Learning for Shruti Assignment

## Toy Problem

I have a swarasamooha. I'm operating within a bounded tonnetz net.  
I want to evaluate a "cost" of each shruti assignment to every note in the samooha.

**Example:**  
My samooha is `b1 = Sgn,Sn,SgSggn,n,n,S` and set of notes are `{S, g, n}` with options within my net `{S, g1/g2, n1/n2}`

I'd like to arrive at a parameterization/family `J` that evaluate `J(b1(S,g1,n1)), J(b1(S,g1,n2)), J(b1(S,g2,n1)), J(b1(S,g2,n2))`

Eventually, I'd like to use this for contrastive learning:  
For 2 samoohas, `b1, b2` if I know for a fact that `{S, g1, n2}` minimizes `J(b1)` and `{S, g2, n1}` minimizes `J(b2)`,  
then I can use this fact to learn `J`

## The nature of the evaluation function

**0th order**
- collapse `b` into a histogram; `J` is not a function of a histogram on notes
- this ignores *movement/momentum*

**arbitrary J**
- ?

## Goal

Eventually, I'd like to use this for contrastive learning:<br>
for 2 samoohas, `b1, b2` if I know for a fact that `{S, g1, n2}` minimizes `J(b1)` and `{S, g2, n1}` minimizes `J(b2)`,<br>
then I can use this fact to learn `J`
