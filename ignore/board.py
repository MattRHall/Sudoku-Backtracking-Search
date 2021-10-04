import numpy as np
import copy

class board():
    
    def __init__(self, board):
        """ Creates board class with attributes final / possibilities"""
        self.start = board.flatten()
        self.final = np.array([-1 if i == 0 else i for i in np.nditer(self.start)])
        
        self.possibilities = np.array([range(1,10) if _ == 0 else [0] * 9 for _ in np.nditer(self.start)])
        for i,v in np.ndenumerate(self.start):
            if v != 0:
                self.possibilities[i[0]][v - 1] = v
            else:
                cells = self.get_col_row_grid(i[0])
                for idx in set(cells[0] + cells[1] + cells[2]):
                    if self.final[idx] != -1:
                        self.possibilities[i[0]][self.final[idx]-1] = 0
                        
    @staticmethod
    def get_col_row_grid(position):
        """ Returns the column / row / grid indices for a given position """
        col_idx = [(9*_)+ position%9 for _ in range(9)]
        row_idx = list(range(position-(position%9),position-(position%9)+9))
        col_pos = position%9 - position%9%3
        row_pos = ((position//9)*9) - ((position//9)*9)%27
        grid_idx = [col_pos + row_pos + _ for _ in range(3)] + \
                    [col_pos + row_pos + _ for _ in range(9,12)] + \
                    [col_pos + row_pos + _ for _ in range(18,21)]
        return col_idx, row_idx, grid_idx
    
    def get_unique_value_count(self):
        unique, counts = np.unique(self.final, return_counts = True)
        return dict(zip(unique, counts))
        
    def is_valid_answer(self):
        """ Returns True if sudoku with 1-9 in rows, 1-9 in columns and 1-9 in boxes """
        rows = [set([self.final[i+(9*j)] for i in range(0,9)]) == set(range(1,10)) for j in range(0,9)]
        cols = [set([self.final[(9*i)+j] for i in range(0,9)]) == set(range(1,10)) for j in range(0,9)]
        grid = [set([self.final[col + row + _] for _ in range(3)] +
                 [self.final[col + row + _] for _ in range(9,12)] +
                 [self.final[col + row + _] for _ in range(18,21)]) == set(range(1,10)) for col in [0,3,6] for row in [0, 27, 54]]

        return self.final.reshape(9,9) if all(rows) == True and all(cols) == True and all(grid) == True else np.full((9,9),-1)
     
    def is_goal(self):
        """ Returns True if all values in the final array are not -1 """
        return True if np.all(self.final != -1) else False
      
    def is_invalid(self):
        """ Returns False if there is an array in possibilities full of 0's """
        return True if np.any([not np.any(i) for i in self.possibilities]) else False
    
    def get_singletons(self):
        """ Returns the positions where we have a 'certain' value but final not updated """
        return [index[0] for index,n in np.ndenumerate(np.count_nonzero(self.possibilities == 0, axis = 1)) if (self.final[index] == -1 and n == 8)]
        
    def get_possible_values(self, position):
        """ Returns the possible value from a given position (excludes 0) """
        arr = self.possibilities[position]
        return arr[np.where(arr != 0)].copy()
    
    def set_value(self, position, value): #position(0-80)
        """ Creates new state, adds in position, checks if further improvements can be made """

        state = copy.deepcopy(self)
        state.possibilities[position] = [0 if i != value else value for i in range(1,10)]
        state.final[position] = value

        # REMOVE FROM ROW / COLUMNS / GRID POSSIBILITIES
        col_row_grid = self.get_col_row_grid(position)
        for pos in set(col_row_grid[0] + col_row_grid[1] + col_row_grid[2]):
            if pos != position:
                if self.final[pos] == -1:
                    state.possibilities[pos][value - 1] = 0
                           
        return self.set_value_loop(state)
    
    def set_value_loop(self, state):
        """ Recrusive loop to turn all singletons into final values (and then reset)"""
        singleton_positions = state.get_singletons()
        while len(singleton_positions) > 0:
            new_pos = singleton_positions[0]
            state = state.set_value(new_pos, state.get_possible_values(new_pos)[0])
            singleton_positions = state.get_singletons()
        
        return state