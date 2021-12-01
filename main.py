import matplotlib.pyplot as plt
import numpy as np
import time


class PlotResults:
    """
    Class to plot the results. 
    """

    def plot_results(self, data1, data2, label1, label2, filename):
        """
        This method receives two lists of data point (data1 and data2) and plots
        a scatter plot with the information. The lists store statistics about individual search 
        problems such as the number of nodes a search algorithm needs to expand to solve the problem.

        The function assumes that data1 and data2 have the same size. 

        label1 and label2 are the labels of the axes of the scatter plot. 
        
        filename is the name of the file in which the plot will be saved.
        """
        _, ax = plt.subplots()
        ax.scatter(data1, data2, s=100, c="g", alpha=0.5, cmap=plt.cm.coolwarm, zorder=10)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.grid()
        plt.savefig(filename)


class Grid:
    """
    Class to represent an assignment of values to the 81 variables defining a Sudoku puzzle. 

    Variable _cells stores a matrix with 81 entries, one for each variable in the puzzle. 
    Each entry of the matrix stores the domain of a variable. Initially, the domains of variables
    that need to have their values assigned are 123456789; the other domains are limited to the value
    initially assigned on the grid. Backtracking search and AC3 reduce the the domain of the variables 
    as they proceed with search and inference.
    """

    def __init__(self):
        self._cells = []
        self._complete_domain = "123456789"
        self._width = 9

    def copy(self):
        """
        Returns a copy of the grid. 
        """
        copy_grid = Grid()
        copy_grid._cells = [row.copy() for row in self._cells]
        return copy_grid

    def get_cells(self):
        """
        Returns the matrix with the domains of all variables in the puzzle.
        """
        return self._cells

    def get_width(self):
        """
        Returns the width of the grid.
        """
        return self._width

    def read_file(self, string_puzzle):
        """
        Reads a Sudoku puzzle from string and initializes the matrix _cells. 

        This is a valid input string:

        4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......

        This is translated into the following Sudoku grid:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        i = 0
        row = []
        for p in string_puzzle:
            if p == '.':
                row.append(self._complete_domain)
            else:
                row.append(p)

            i += 1

            if i % self._width == 0:
                self._cells.append(row)
                row = []

    def print(self):
        """
        Prints the grid on the screen. Example:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        for _ in range(self._width + 4):
            print('-', end=" ")
        print()

        for i in range(self._width):

            print('|', end=" ")

            for j in range(self._width):
                if len(self._cells[i][j]) == 1:
                    print(self._cells[i][j], end=" ")
                elif len(self._cells[i][j]) > 1:
                    print('.', end=" ")
                else:
                    print(';', end=" ")

                if (j + 1) % 3 == 0:
                    print('|', end=" ")
            print()

            if (i + 1) % 3 == 0:
                for _ in range(self._width + 4):
                    print('-', end=" ")
                print()
        print()

    def print_domains(self):
        """
        Print the domain of each variable for a given grid of the puzzle.
        """
        for row in self._cells:
            print(row)

    def is_solved(self):
        """
        Returns True if the puzzle is solved and False otherwise. 
        """
        for i in range(self._width):
            for j in range(self._width):
                if len(self._cells[i][j]) != 1:
                    return False
        return True


class VarSelector:
    """
    Interface for selecting variables in a partial assignment. 

    Extend this class when implementing a new heuristic for variable selection.
    """

    def select_variable(self, grid):
        pass


class FirstAvailable(VarSelector):
    """
    Naïve method for selecting variables; simply returns the first variable encountered whose domain is larger than one.
    """

    def select_variable(self, grid):
        # Go through the grid and get back the first variable with a domain > 1
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):

                if len(grid.get_cells()[i][j]) > 1:
                    return [i, j]


class MRV(VarSelector):
    """
    Implements the MRV heuristic, which returns one of the variables with smallest domain. 
    """

    def select_variable(self, grid):
        # Go through the grid and get the variable with the smallest domain that's > 1

        # min_domain = [length of domain, i, j]
        min_domain = [9, 0, 0]

        for i in range(grid.get_width()):
            for j in range(grid.get_width()):

                if len(grid.get_cells()[i][j]) == 2:
                    # If the domain is 2, just return it immediately, we won't find a smaller one and tie-breaking doesn't matter
                    return [i, j]

                if 2 < len(grid.get_cells()[i][j]) < min_domain[0]:
                    min_domain = [len(grid.get_cells()[i][j]), i, j]

        return [min_domain[1], min_domain[2]]

class AC3:
    """
    This class implements the methods needed to run AC3 on Sudoku. 
    """

    def remove_domain_row(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same row. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != column:
                new_domain = grid.get_cells()[row][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[row][j]) > 1:
                    variables_assigned.append((row, j))

                grid.get_cells()[row][j] = new_domain

        return variables_assigned, False

    def remove_domain_column(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same column. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != row:
                new_domain = grid.get_cells()[j][column].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[j][column]) > 1:
                    variables_assigned.append((j, column))

                grid.get_cells()[j][column] = new_domain

        return variables_assigned, False

    def remove_domain_unit(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same unit. 
        """
        variables_assigned = []

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue

                new_domain = grid.get_cells()[i][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[i][j]) > 1:
                    variables_assigned.append((i, j))

                grid.get_cells()[i][j] = new_domain
        return variables_assigned, False

    def pre_process_consistency(self, grid):
        """
        This method enforces arc consistency for the initial grid of the puzzle.
        """
        Q = list()

        for i in range(grid.get_width()):
            for j in range(grid.get_width()):

                if len(grid.get_cells()[i][j]) == 1:
                    # Add to Q all variables with a solution locked in (domain = 1)
                    Q.append([i, j])

        self.consistency(grid, Q)

    def consistency(self, grid, Q):
        """
        This is a domain-specific implementation of AC3 for Sudoku.
        """
        # Implement here the domain-dependent version of AC3.

        while Q:
            test = Q.pop()

            # Prune the domains by removing trivial entries
            assigned_row, row_failure = self.remove_domain_row(grid, test[0], test[1])
            assigned_column, column_failure = self.remove_domain_column(grid, test[0], test[1])
            assigned_unit, unit_failure = self.remove_domain_unit(grid, test[0], test[1])

            # Add to Q any variables that are newly locked-in solutions (domain = 1)
            if assigned_row:
                for i in assigned_row:
                    Q.append(i)
            if assigned_column:
                for j in assigned_column:
                    Q.append(j)
            if assigned_unit:
                for k in assigned_unit:
                    Q.append(k)

            # Check if a solution is impossible (Any variables with domain = 0)
            if row_failure or column_failure or unit_failure:
                return True

        return False


class Backtracking:
    """
    Class that implements backtracking search for solving CSPs. 
    """

    def search(self, grid, var_selector):
        """
        Implements backtracking search with inference. 
        """
        if grid.is_solved():
            return grid

        # Get the next variable to be tested
        test = var_selector.select_variable(grid)

        # Check every value in the variable's domain
        for d in grid.get_cells()[test[0]][test[1]]:
            s = [test[0], test[1]]

            copy = grid.copy()
            copy.get_cells()[test[0]][test[1]] = d

            # Check if that value is valid (if ac.consistency returns true, then that value renders a domain 0)
            error = self.check_value(copy, s)
            if not error:
                rb = self.search(copy, var_selector)
                if rb is not False:
                    return rb
        return False

    def check_value(self, grid, test):
        # Implement here the domain-dependent version of AC3.
        ac = AC3()

        # Prune the domains by removing trivial entries
        assigned_row, row_failure = ac.remove_domain_row(grid, test[0], test[1])
        assigned_column, column_failure = ac.remove_domain_column(grid, test[0], test[1])
        assigned_unit, unit_failure = ac.remove_domain_unit(grid, test[0], test[1])

        # Check if a solution is impossible (Any variables with domain = 0)
        if row_failure or column_failure or unit_failure:
            return True

        return False


file = open('tutorial_problem.txt', 'r')
# file = open('top95.txt', 'r')
problems = file.readlines()

running_time_mrv = list()
running_time_first_available = list()

backtracking = Backtracking()
mrv = MRV()
fa = FirstAvailable()
ac = AC3()

# Run in first-available
for p in problems:
    start_time = time.time()
    g = Grid()
    g.read_file(p)
    g.print()
    ac.pre_process_consistency(g)
    g = backtracking.search(g, fa)
    g.print()
    running_time_first_available.append(time.time() - start_time)

# Run with MRV
for p in problems:
    start_time = time.time()
    g = Grid()
    g.read_file(p)
    g.print()
    ac.pre_process_consistency(g)
    g = backtracking.search(g, mrv)
    g.print()
    running_time_mrv.append(time.time() - start_time)

plotter = PlotResults()
plotter.plot_results(running_time_mrv, running_time_first_available,
                     "Running Time Backtracking (MRV)",
                     "Running Time Backtracking (FA)", "running_time")
