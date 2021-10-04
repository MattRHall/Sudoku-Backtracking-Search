# Sudoku Solver
------
### Sudoku Introduction
A standard sudoku puzzle presents a 9x9 playing board which is partially filled with integers [1,9]. In order to solve the puzzle integers [1,9] must be placed in unfilled spaces so that all rows / columns and 3x3 grids each contain all of the integers [1,9].

### Environment / Observations
There is no 'best' search algorithm. Instead different approaches are best suited to different situations. Therefore, it is important to fully understand the situation presented to select the right one for the task at hand. With respect to Sudoku I would make the following observations.   

- **We want the goal state not the path** - We only care about the solution to the puzzle, not the route we took to get there. Therefore we do not need an algorithm that record changes in state over time.

- **There is usually only one solution to a sudoku puzzle** - This means we require our algorithm to be complete, i.e. if a solution does exist then it will be found. An approximate solution is also insufficient.

- **Brute force is not an option** - There are a huge number of possibilities to consider. With no constraints, we would need to consider 9^81 combinations, which would take years. 

- **Sudoku has constraints** - There are 3 constraints that a valid sudoku solution must satisfy. These constraints will drastically reduce the search space, if we choose an appropriate algorithm.

- **The environment is fully observable and deterministic** - We can observe all of the sudoku board at anytime, and we know the impact that adding an integer in a square will have.

### Formal Problem Definition
Before we consider algorithms, it is useful to properly define the problem. This will help us select the most appropriate approach and provide guidance when implementing an algorithm.

- **Variables** - {X<sub>1</sub>...X<sub>n</sub>}, n = integers between [0, 80]. Each variable represents a square in the sudoku board, starting from top left (X<sub>0</sub>) and finishing bottom right (X<sub>80</sub>), moving along a row from left to right, before starting a new row.

- **Domain** - Each variable has the domain {1,2,3,4,5,6,7,8,9} which represent the possible integers that can be placed in a square on a sudoku board.

- **Constraints** - We have 3 global constraints; Alldiff (row), Alldiff (column), Alldiff (grid). Where Alldiff requries all variables to have different values.

### Algorithm Choice
As outlined above, our algorithm must make use of the sudoku constraints because uninformed / brute search methods would lead to huge search spaces that would be impractical to search. The options we will consider for Sudoku are local search / backtracking search (both of which make use of constraints).

- **Local search** - Local search takes a single state and moves to another state by changing one variable at a time. It requires a heuristic to measure whether the state is improved or not. A logical heuristic would be to count the number of constraint violations (column / row / grid) and select the minimum. The main disadvantage with using local search is the risk of local plateaus and local minimums. This leads the risk of not reaching the goal (which is not acceptable). We would also need to keep track of visited states to ensure we didn't cycle back to a already visited state. 

- **Backtracking search** - Backtracking search will repeatedly choose an unassigned variable, try different values, before extending them in a recursive manner. We can further shrink the search space by eliminating / pruning values (states) that are inconsistent (forward checking). This seems the most suitable approach. We can also explore which heuristics work best, and look at adding advanced sudoku techniques when applying forward checking. 
 
### Algorithm - Backtracking Search with Inference
The search algorithm we will apply to the sudoku puzzle is **backtracking search** with inference (**forward checking**). Forward checking will apply arc consistency to all variables, to ensure no variables are inconsistent with each other. We will also apply **minimum remaining values** when selecting variables. Pseudocode for the approach is laid out below. 

* Constraint-Propagation
    -    Run constraint propagation (ARC consistency), return state.  
    - If state is a final solution, **return** answer if valid, else return array of -1's.  

* Run backtracking search
    - Select variable with lowest number of potential values in its domain.  
    - Select the lowest value in the domain of this variable.  
    - Set the variable to this value.  
    - Apply constraint propagation (ARC consistency) to remaining variables.  
    - If state is a final solution, **return** answer if valid, else return array of -1's.  
    - If state is consistent (but not complete) then repeat backtracking search.  

### Algorithm - Failed Enchancements
**Selecting value by most common occurence** - By focusing on the most common integers on the sudoku board that have been finalized it might be possible to find solutions more quickly. Although it did work in some instances, the computational cost of building and sorting a dictionary in every iteration was too great.
```
def get_unique_value_count(self):
        unique, counts = np.unique(self.final, return_counts = True)
        return dict(zip(unique, counts))

common_values = partial_state.get_unique_value_count()
ordered_values = []
for temp_value in values:
    ordered_values.append((temp_value, common_values.get(temp_value,0)))
    ordered_values.sort(key = lambda x: x[1], reverse = True)
```
**Naked pairs** - The objective here was to improve on the constraint propagation by looking for naked pairs, i.e. if two variables in a row/colum/grid can only be {1,5} then none of the other variables can be {1,5}. Disappointingly this failed to make a meaningful improvement. Although it did improve in some instances, the computational cost was too great. As a result of this I didn't build naked triples or other sudoku solving techniques (naked pairs should be the most beneficial).
```
    def naked_pair(self, positions): #positions = array of 9 positions
        naked_pairs = []
        for pos1 in positions:
            for pos2 in positions:
                if pos1 != pos2:
                    if self.final[pos1] == -1 and self.final[pos2] == -1:
                        if len(self.get_possible_values(pos1)) == 2 and len(self.get_possible_values(pos1)) == 2:
                            if all(self.possibilities[pos1] == self.possibilities[pos2]):
                                if (pos1, pos2) not in naked_pairs and (pos2, pos1) not in naked_pairs:
                                    naked_pairs.append((pos1, pos2))
```

### Further work
Further work would focus on improved implementation of the algorithm (identify bottlenecks, and small changes to data structure choices) that could lead to improvements in speed. We would also try using lists as data structures instead of numpy arrays (given no matrix multiplication / operations are actually performed). Finally, we would explore additional heuristics as the choice of the heuristic could make a big difference to computational time (e.g. maximum cardinality ordering, minimum bandwidth ordering).

