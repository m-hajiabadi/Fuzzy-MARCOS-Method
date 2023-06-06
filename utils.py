import numpy as np
import matplotlib.pyplot as plt

class FuzzyNum():
    def __init__(self, l, m , u):
        self.l = l
        self.m = m
        self.u = u

    def __str__(self):
        return f'(l:{self.l}, m:{self.m}, u:{self.u})'
    
    def reciprocal(self):
        return (self.l **-1, self.m**-1, self.u**-1)

def addition(A1, A2):
    return FuzzyNum(A1.l + A2.l, A1.m + A2.m, A1.u + A2.u)

def multiplication(A1, A2):
    return FuzzyNum(A1.l * A2.l, A1.m * A2.m, A1.u * A2.u)

def division(A1, A2):
    return FuzzyNum(A1.l / A2.u, A1.m / A2.m, A1.u / A2.l)

def subtraction(A1, A2):
    return FuzzyNum(A1.l - A2.u, A1.m - A2.m, A1.u - A2.l)

import numpy as np

def process_fuzzy_file(file_path):
    # Open the file
    with open(file_path, 'r') as file:
        # Read the lines of the file
        lines = file.readlines()

    # Extract the criteria from the first line
    criteria = lines[0].strip().split()

    criteria_dict = dict(enumerate(criteria))
    # Initialize an empty list to store the fuzzy numbers
    FuzzyNumbers = []

    # Process each line (excluding the first line)
    for line in lines[1:]:
        # Remove leading and trailing whitespace
        line = line.strip()
        # Split the line by whitespace to get individual fuzzy numbers
        numbers = line.split()
        # Convert each fuzzy number from string to FuzzyNum object
        fuzzy_row = [FuzzyNum(*(map(int, num.strip('()').split(',')))) for num in numbers]
        # Append the row of fuzzy numbers to the list
        FuzzyNumbers.append(fuzzy_row)

    # Convert the list of fuzzy numbers to a NumPy array
    fuzzy_array = np.array(FuzzyNumbers)

    return fuzzy_array, criteria_dict

def process_criteria_weights_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    scenarios = []
    scenario = []
    for line in lines:
        line = line.strip()
        if line.startswith("S"):
            if scenario:
                scenarios.append(scenario)
                scenario = []
        else:
            l, m, u = tuple(map(float, line.strip("()").split(",")))
            weight = FuzzyNum(float(l), float(m), float(u))
            scenario.append(weight)
    scenarios.append(scenario)

    return scenarios


def create_extended_fuzzy_matrix(fuzzy_array, criteria_dict):
    
    # Calculate Ã(AI) and Ã(ID) based on Equations (20) and (21)
    ai_solution = []
    id_solution = []
    # print(range(fuzzy_array.shape[1]))
    for column_index in range(fuzzy_array.shape[1]):
        column_values = fuzzy_array[:, column_index]
        # Calculate Ã(AI)
        ai_value_l = min(
            num.l
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        ai_value_m = min(
            num.m
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        ai_value_u = min(
            num.u
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        ai_solution.append(FuzzyNum(ai_value_l, ai_value_m, ai_value_u))

        # Calculate Ã(ID)
        id_value_l = max(
            num.l
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        id_value_m = max(
            num.m
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        id_value_u = max(
            num.u
            for num in column_values
            if criteria_dict[column_index] in ['B', 'C']
        )
        id_solution.append(FuzzyNum(id_value_l, id_value_m, id_value_u))

    # Convert the Ã(AI) and Ã(ID) solutions to arrays
    ai_solution_array = np.array([ai_solution])
    id_solution_array = np.array([id_solution])

    return np.concatenate((ai_solution_array, fuzzy_array, id_solution_array))

def create_normalized_fuzzy_matrix(extended_matrix, criteria_dict):
    # Extract id_solution and ai_solution from the extended matrix
    id_solution = extended_matrix[0, :]
    ai_solution = extended_matrix[-1, :]
    # Create an empty matrix to store the normalized fuzzy values
    normalized_matrix = np.empty_like(extended_matrix)

    # Process each element in the extended matrix
    for i in range(extended_matrix.shape[0]):
        for j in range(extended_matrix.shape[1]):
            value = extended_matrix[i, j]

            # Check if the column criterion is in C
            if criteria_dict[j] == 'C':
                # Apply Equation (22) for normalization
                nl = id_solution[j].l / value.u
                nm = id_solution[j].l / value.m
                nu = id_solution[j].l / value.l
            # Check if the column criterion is in B
            elif criteria_dict[j] == 'B':
                # Apply Equation (23) for normalization
                nl = value.l / ai_solution[j].u
                nm = value.m / ai_solution[j].u
                nu = value.u / ai_solution[j].u
            else:
                # For criteria not in B or C, set the normalized values to 0
                nl = nm = nu = 0

            # Store the normalized values in the matrix
            normalized_matrix[i, j] = FuzzyNum(nl, nm, nu)

    return normalized_matrix


def compute_weighted_fuzzy_matrix(normalized_matrix, criteria_weights):
    # Compute the weighted fuzzy matrix
    weighted_matrix = np.empty_like(normalized_matrix)
    # Process each element in the normalized matrix
    for i in range(normalized_matrix.shape[0]):
        for j in range(normalized_matrix.shape[1]):
            value = normalized_matrix[i, j]
            weight = criteria_weights[j]

            # Apply the multiplication operation ⊗ for the fuzzy values and weights
            vl = value.l * weight.l
            vm = value.m * weight.m
            vu = value.u * weight.u

            # Store the weighted values in the matrix
            weighted_matrix[i, j] = FuzzyNum(vl, vm, vu)

    return weighted_matrix


def calculate_fuzzy_matrix_S_i(weighted_matrix):
    S_i = []
    for row in weighted_matrix:
        s_i = FuzzyNum(0, 0, 0)
        for j in range(len(row)):
            s_i = addition(s_i, row[j])
        S_i.append(s_i)

    return S_i

def max_ti(t_i):
    ti_value_l = max(
        num.l
        for num in t_i[1:-1]
    )
    ti_value_m = max(
        num.m
        for num in t_i[1:-1]
    )
    ti_value_u = max(
        num.u
        for num in t_i[1:-1]
    )
    return FuzzyNum(ti_value_l, ti_value_m, ti_value_u)



     
def calculate_utility_degree_minus(S_i, S_ai):
    return [division(s_i, S_ai) for s_i in S_i]


def calculate_utility_degree_plus(S_i, S_id):
    return [division(s_i, S_id) for s_i in S_i]

def calculate_fuzzy_matrix_t_i(K_i_minus, K_i_plus):
    return [addition(K_i_minus[i], K_i_plus[i]) for i in range(len(K_i_minus))]


def defuzzify_fuzzy_number(fuzzy_number):
    return (fuzzy_number.l + 4 * fuzzy_number.m + fuzzy_number.u) / 6


def defuzzify(fuzzy_number, df_crisp):
    # df_crisp = defuzzify_fuzzy_number(fuzzy_number)
    return FuzzyNum(fuzzy_number.l / df_crisp, fuzzy_number.m / df_crisp, fuzzy_number.u / df_crisp)

def calculate_utility_functions(K_i_minus, K_i_plus, df_crisp):
    f_K_i_plus = []
    f_K_i_minus = []
    
    for i in range(len(K_i_minus)):
        f_K_i_plus.append(defuzzify(K_i_minus[i], df_crisp))   # defuzzify K_i_minus
        f_K_i_minus.append(defuzzify(K_i_plus[i], df_crisp))   # defuzzify K_i_plus

    return f_K_i_plus, f_K_i_minus


def calculate_utility_function_alternatives(f_K_i_plus, f_K_i_minus, K_i_plus, K_i_minus):
    f_K_i = []

    for i in range(len(f_K_i_plus)):
        num = defuzzify_fuzzy_number(K_i_plus[i])+ defuzzify_fuzzy_number(K_i_minus[i])
        denom = 1 + (1 - defuzzify_fuzzy_number(f_K_i_plus[i])) / defuzzify_fuzzy_number(f_K_i_plus[i]) + (1 - defuzzify_fuzzy_number(f_K_i_minus[i])) / defuzzify_fuzzy_number(f_K_i_minus[i])
        utility = num/denom
        f_K_i.append(utility)

    return f_K_i


def compute_ranks(f_K_i):
    # Assign rank to each alternative
    sorted_indices = np.argsort(-f_K_i)

    # Initialize ranks
    ranks = np.zeros_like(sorted_indices)

    # Assign ranks based on the sorted indices, considering ties
    rank = 1
    for i in range(len(sorted_indices)):
        if i > 0 and f_K_i[sorted_indices[i]] != f_K_i[sorted_indices[i-1]]:
            rank += 1
        ranks[sorted_indices[i]] = rank
    return ranks

def plot_scenarios(all_ranks, alternatives):

    avg_line = np.mean(all_ranks, axis=0)
    
    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Calculate the positions for the bars
    positions = alternatives + np.arange(len(all_ranks)).reshape(-1, 1) * 0.1

    # Set color cycle for line plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_ranks)))

    # Plot the bar diagrams for all_ranks_i
    for i, (f_K_i, color) in enumerate(zip(all_ranks, colors)):
        ax.bar(positions[i]+i, f_K_i, color=color, alpha=0.5, width=2, align='center', label=f'f_K_{i+1}')
        
        # Plot line plot for each f_K_i
        ax.plot(positions[i], avg_line, color=color, linewidth=2)

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Alternatives')
    ax.set_ylabel('Rank')

    # Set the chart title
    ax.set_title('Bar Diagrams and Line Plots')

    # Add a legend
    ax.legend()

    plt.savefig('output.png')
    # Show the plot
    plt.show()

