1. Parse the table of all odav raags
2. Construct an adjacency graph.
	- each node is a raag
	- two raags are neighbors if their strings are one edit distance away
	- e.g. SRGPD has neighbor SRmPD (and also SRMPD even though G and M are two steps away)
	- however, SRGDN is NOT a neighbor, since the placement of D has also changed (it's not just editing a note)
3. Find all cycles present in this graph.
4. Find all _spanning_ cycles present in this graph.
