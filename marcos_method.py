import numpy as np
from utils import *


def main():
    
    # Step 1:
    # Convert the list of fuzzy numbers to a NumPy array
    fuzzy_array, criteria_dict = process_fuzzy_file('data/fuzzy_numbers.txt')
    
    # Step 2:
    extended_matrix = create_extended_fuzzy_matrix(fuzzy_array=fuzzy_array, criteria_dict=criteria_dict)
    
    # Step 3:
    normalized_matrix = create_normalized_fuzzy_matrix(extended_matrix=extended_matrix, criteria_dict=criteria_dict)
    
    # Step 4:
    # Read fuzzy weight coefficients from file
    scenarios = process_criteria_weights_file('data/criteria_weights.txt')
    
    all_ranks = []
    for criteria_weights in scenarios:
        
        weighted_matrix = compute_weighted_fuzzy_matrix(normalized_matrix=normalized_matrix, criteria_weights=criteria_weights)
        # Step 5:
        # for i in weighted_matrix:
        #     print(i[0])
        S_i = calculate_fuzzy_matrix_S_i(weighted_matrix=weighted_matrix)
        
        S_ai = S_i[0]  # Assuming S_ai is the first element of S_i
        S_id = S_i[-1]  # Assuming S_id is the last element of S_i
        
        # Step 6:
        K_i_minus = calculate_utility_degree_minus(S_i=S_i, S_ai=S_ai)
        K_i_plus = calculate_utility_degree_plus(S_i=S_i, S_id=S_id)
        # Step 7:
        t_i = calculate_fuzzy_matrix_t_i(K_i_minus=K_i_minus, K_i_plus=K_i_plus)
        
        # D = max(t_i, key=lambda x: x.m)
        D = max_ti(t_i)
        df_crisp = defuzzify_fuzzy_number(D)
        
        # Step 8:
        # Calculate utility functions
        f_K_i_plus, f_K_i_minus = calculate_utility_functions(K_i_minus, K_i_plus, df_crisp)
        
        
        # Step 9:
        # Calculate utility function of alternatives
        f_K_i = calculate_utility_function_alternatives(f_K_i_plus, f_K_i_minus, K_i_plus, K_i_minus)
        # Remove first(anti ideal) and last(ideal) row from f_K_i
        f_K_i = np.array(f_K_i[1:-1])
        
        ranks = compute_ranks(f_K_i=f_K_i)
            
        all_ranks.append(ranks)
        
    # Set the horizontal axis as Alternatives
    alternatives = list(range(len(f_K_i)))
    alternatives = [alter*20 for alter in alternatives]
    
    # Plot all scenarios
    plot_scenarios(all_ranks=all_ranks, alternatives=alternatives)
    
if __name__ == "__main__":
    main()