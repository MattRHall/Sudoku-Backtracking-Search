# Sudoku Solver
------
## Sudoku Introduction
A standard sudoku puzzle presents a 9x9 playing board which is partially filled with integers [1,9]. In order to solve the puzzle integers [1,9] must be placed in unfilled spaces so that all rows / columns and 3x3 grids each contain all of the integers [1,9].

## Environment / Observations
There are a number of features about a sudoku puzzle that helps to guide algorithm choice: 

- **We want the goal state not the path** - We only care about the solution to the puzzle, not the route we took to get there. Therefore we do not need an algorithm that record changes in state over time.

- **There is usually only one solution to a sudoku puzzle** - This means we require our algorithm to be complete, i.e. if a solution does exist then it will be found. An approximate solution is also insufficient.

- **Brute force is not an option** - There are a huge number of possibilities to consider. With no constraints, we would need to consider 9^81 combinations, which would take years. 

- **Sudoku has constraints** - There are 3 constraints that a valid sudoku solution must satisfy. These constraints will drastically reduce the search space, if we choose an appropriate algorithm.

- **The environment is fully observable and deterministic** - We can observe all of the sudoku board at anytime, and we know the impact that adding an integer in a square will have.

## Algorithm - Backtracking Search with Forward Checking
Backtracking search with forward checking is similar to depth-first search. However the search tree is reduced by applying sudoku constraints (also call ARC consistency), to remove states that can't be a solution. Furthermore, the cells with only a small number of possibilities are searched first, to identify and prune states that can't be solutions quickly. For example, if cell-1 can be {1,3} and cell-2 can be {1,5,7,8} then we should search the first cell first. 

Additional work was done to explore heuristics that could be applied to improve performance by pruning the search tree. This includes searching based on most common integers, and applying naked pairs (f two variables in a row/colum/grid can only be {1,5} then none of the other variables can be {1,5}). However, the increased computational cost of these heuristics did not offset the benefit of a smaller search tree. There are many other different sudoku solving techniques, but checking whether they are applicable at each possible state precludes their use.

## Usage
The sudoku class, has the method **sudoku_solver()** which takes a numpy array (shape=9x9) as an input. The numpy array should have 0's in place of unknown values, and values between 1-9 elsewhere. If there is no solution to the sudoku, or the input is invalid, then a numpy array (shape=9x9) full of -1's
 
```
puzzle = np.array([[0, 8, 5, 0, 1, 3, 0, 0, 9],
       [6, 3, 4, 0, 0, 2, 1, 7, 5],
       [0, 2, 0, 5, 7, 4, 0, 3, 0],
       [2, 4, 8, 3, 6, 7, 9, 5, 1],
       [9, 6, 0, 4, 5, 8, 0, 2, 3],
       [3, 5, 7, 2, 0, 0, 4, 8, 0],
       [5, 7, 3, 1, 0, 0, 8, 9, 2],
       [4, 9, 6, 0, 2, 5, 3, 1, 0],
       [8, 1, 2, 0, 3, 9, 5, 6, 4]])

game = sudoku()
game.sudoku_solver(puzzle)
```

## Classes
(class) **Board** This class contains the functionality of a sudoku board. It is an inner class of the Sudoku class.  
- Key attributes:  
    - **final**: A Numpy array of shape 1x81 containing either finalized sudoku values or -1's.
    - **possibilities**: A numpy array of shape 89x9 containing the possible values for each cell.
- Key methods:
    - **_set_value(position, value)**: Places a number in a cell, and removes possibilities from other cells if constraints not met.
    - **_get_singletons()**: Finds cells where there is only one possible value, but the board has not been updated for it yet.
    - **_is_goal()**: Returns True if a solution has been found (i.e. all values in *final* are not -1)
    - **_get_col_row_grid(position)**: Returns the row / col / grid (3x3) indices for a given cell.

(class) **Sudoku** This class contains the functionality of the search algorithm. 
- Key methods:
    - **_order_values(partial_state, position)**: Retrieves a list of cells with more than one possibilities, ordered by smallest number of possibilities to largest.
    - **_depth_first_search(partial_state)**: Performs depth first search, applying constraints at each stage to prune search tree.
    - **sudoku_solver(sudoku)**: Checks for solution by applying constraints iteratively. If no solution found, *_depth_first_search* applied.

