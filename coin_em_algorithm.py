# Github : EonTechie

import random
from math import comb

# Calculates the probability of getting a certain number of 'Heads' in a sequence using the binomial distribution.
def binomial_probability(heads, flips, bias_head): # heads: Number of 'H' observed , flips: Total number of flips (e.g., 10 flips) , bias_head: Probability of getting 'H' in a single flip (e.g., 0.5)  
    # Likelihood calculation using the binomial formula
    likelihood = comb(flips, heads) * (bias_head ** heads) * ((1 - bias_head) ** (flips - heads))
    return likelihood

def generate_coin_data(num_sequences=10, prob_choose_a=0.5, a_bias=0.6, b_bias=0.5):
    """
    Generate coin flip sequences based on known biases and priors.
    
    Parameters:
        num_sequences (int): Number of sequences to generate.
        prob_choose_a (float): Prior probability of choosing Coin A.
        a_bias (float): Bias (probability of heads) for Coin A.
        b_bias (float): Bias (probability of heads) for Coin B.

    Returns:
        list: List of tuples with ('unknown', sequence).
    """
    data = []
    for _ in range(num_sequences):
        # Decide the coin based on prior probabilities
        coin_bias = a_bias if random.random() < prob_choose_a else b_bias  # Select Coin A or Coin B based on prior
        sequence = ''.join(['H' if random.random() < coin_bias else 'T' for _ in range(10)])  # Generate flips
        data.append(('unknown', sequence))  # Append with 'unknown' label
    return data

def count_heads_and_tails(data):
    """
    Count the number of heads and tails in each sequence.

    Parameters:
        data (list): List of tuples with coin labels and sequences.

    Returns:
        list: Processed results with counts of heads and tails.
    """
    results = []
    for coin, sequence in data:
        heads_count = sequence.count('H')  # Count the number of H in the sequence
        tails_count = sequence.count('T')  # Count the number of T in the sequence
        results.append({
            'coin': coin,
            'sequence': sequence,
            'heads_count': heads_count,
            'tails_count': tails_count
        })
    return results

def em_algorithm(sequences, max_iter=100, tol=1e-6):
    """
    Expectation-Maximization (EM) algorithm to estimate parameters of coin biases.

    Parameters:
        sequences (list): List of sequences of coin flips.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        tuple: Estimated biases (Q_A, Q_B) and posterior probabilities (Posterior_A, Posterior_B).
    """
    # Initial parameters
    Q_A = 0.6  # Probability of Heads for Coin A
    Q_B = 0.5  # Probability of Heads for Coin B
    P_A = 0.5  # Prior probability of choosing Coin A
    P_B = 0.5  # Prior probability of choosing Coin B
    
    for iteration in range(max_iter):
        # E Step: Calculate probabilities of belonging to Coin A and Coin B
        Posterior_A = []
        Posterior_B = []
        
        print(f"Iteration: {iteration+1}")
        print("- Expectation Step\n") 
        
        for sequence in sequences:
            heads = sequence.count('H')
            tails = sequence.count('T')
            flips = heads + tails

            # Calculate likelihoods (including binomial coefficients)
            joint_probability_A = binomial_probability(heads, flips, Q_A) * P_A
            joint_probability_B = binomial_probability(heads, flips, Q_B) * P_B
            
            # Normalize to find Q values
            total_probability = joint_probability_A + joint_probability_B
            Posterior_A.append(joint_probability_A / total_probability)
            Posterior_B.append(joint_probability_B / total_probability)
            
            print(f"Sequence: {sequence}")
            
            # Likelihood
            print(f"  Likelihood (A): P(S | A) = {binomial_probability(heads, flips, Q_A):.4f}, Likelihood (B): P(S | B) = {binomial_probability(heads, flips, Q_B):.4f}")

            # Joint Probability (Likelihood × Prior)
            print(f"  Joint Probability (A): P(S, A) = {joint_probability_A:.4f} ,  Joint Probability (B): P(S, B) = {joint_probability_B:.4f} ")
            # Posterior Probability (Normalized)
            print(f"  Posterior Probability: P(A | S) = {joint_probability_A / total_probability:.4f}, P(B | S) = {joint_probability_B / total_probability:.4f}")

            # Number of Heads attributed to A and B
            print(f" #Heads Attributed to A : {sequence.count('H')} * P(A | S) = {sequence.count('H')*(joint_probability_A / total_probability)}, #Heads Attributed to B = {(sequence.count('H')*joint_probability_B / total_probability)}\n")
            
        print("Maximization Step:\n")
        
        # M Step: Update Q_A and Q_B (calculate heads attributed to A and heads attributed to B , and update bias)
        updated_Q_A = sum(Posterior_A[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                    sum(Posterior_A[i] * len(sequences[i]) for i in range(len(sequences)))

        updated_Q_B = sum(Posterior_B[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                    sum(Posterior_B[i] * len(sequences[i]) for i in range(len(sequences)))
        
        # Print Updated biases
        print(f"Updated Q_A:{updated_Q_A}\nUpdated Q_B {updated_Q_B}\n")
        
        # Print convergence constraint
        print(f"Change_in_QA:{abs(updated_Q_A - Q_A)}\nChange_in_QB: {abs(updated_Q_B - Q_B)}\n")
        
        # Stopping criteria
        if abs(updated_Q_A - Q_A) < tol and abs(updated_Q_B - Q_B) < tol:
            break

        # Update parameters
        Q_A = updated_Q_A
        Q_B = updated_Q_B

    return Q_A, Q_B, Posterior_A, Posterior_B

# Generate 10 sequences of coin flips
data = generate_coin_data(10)  # Generate 10 sets of data
results = count_heads_and_tails(data)
sequences = [result['sequence'] for result in results]

# Apply EM algorithm to estimate parameters
Q_A, Q_B, Posterior_A, Posterior_B = em_algorithm(sequences)

# Final results after convergence
print("After all EM steps implemented until it converges:\n")
print(f"Final Q_A (Head Probability for Coin A): {Q_A:.4f}\n")
print(f"Final Q_B (Head Probability for Coin B): {Q_B:.4f}\n")

# Determine which coin each sequence belongs to using the last found values after EM algorithm
assignments = []  # List to store sequence assignments to coins
for i, sequence in enumerate(sequences):
    if Posterior_A[i] > Posterior_B[i]:  # If Q_A value is greater, assign to Coin A
        assignments.append(('A', sequence))
    else:  # Otherwise, assign to Coin B
        assignments.append(('B', sequence))

# Print sequence assignments to coins
for coin, sequence in assignments:
    print(f"Sequence: {sequence} is likely to be Coin {coin}")
