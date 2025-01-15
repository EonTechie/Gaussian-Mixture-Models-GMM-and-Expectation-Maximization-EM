# GitHub : EonTechie

import random
from math import comb


# Calculates the probability of getting a certain number of 'Heads' in a sequence using the binomial distribution.
def binomial_probability(heads, flips, bias_head):
    likelihood = comb(flips, heads) * (bias_head ** heads) * ((1 - bias_head) ** (flips - heads))
    return likelihood


def generate_coin_data(num_sequences=10, prob_choose_a=0.9, a_bias=0.6, b_bias=0.5):
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
    random.seed(42)  # Set seed for reproducibility
    data = []
    for _ in range(num_sequences):
        # Decide the coin based on prior probabilities
        coin_bias = a_bias if random.random() < prob_choose_a else b_bias  # Select Coin A or Coin B based on prior
        sequence = ''.join(['H' if random.random() < coin_bias else 'T' for _ in range(10)])  # Generate flips
        data.append(('unknown', sequence))  # Append with 'unknown' label
    return data


def count_heads_and_tails(data):
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
    # Initial parameters with unequal priors
    Q_A = 0.6  # Probability of Heads for Coin A
    Q_B = 0.5  # Probability of Heads for Coin B
    P_A = 0.7  # Prior probability of choosing Coin A (unequal initialization)
    P_B = 0.3  # Prior probability of choosing Coin B (unequal initialization)
    
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

            # Calculate joint probabilities (Likelihood × Prior)
            joint_probability_A = binomial_probability(heads, flips, Q_A) * P_A
            joint_probability_B = binomial_probability(heads, flips, Q_B) * P_B
            
            # Normalize to calculate posterior probabilities
            total_probability = joint_probability_A + joint_probability_B
            Posterior_A.append(joint_probability_A / total_probability)
            Posterior_B.append(joint_probability_B / total_probability)
            
            print(f"Sequence: {sequence}")
            print(f"  Joint Probability (A): P(S, A) = {joint_probability_A:.4f}, Joint Probability (B): P(S, B) = {joint_probability_B:.4f}")
            print(f"  Posterior Probability: P(A | S) = {Posterior_A[-1]:.4f}, P(B | S) = {Posterior_B[-1]:.4f}\n")

        print("Maximization Step:\n")
        
        # M Step: Update Q_A and Q_B (update biases)
        
        updated_Q_A = sum(Posterior_A[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                      sum(Posterior_A[i] * len(sequences[i]) for i in range(len(sequences)))
        updated_Q_B = sum(Posterior_B[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                      sum(Posterior_B[i] * len(sequences[i]) for i in range(len(sequences)))
        
        # M Step: Update P_A and P_B (update priors)
        # M Step: Update Prior Probabilities
        
        updated_P_A = (sum(Posterior_A[i] * sequences[i].count('H') for i in range(len(sequences))) + sum(Posterior_A[i] * sequences[i].count('T') for i in range(len(sequences))) ) / (sum(Posterior_B[i] * len(sequences[i]) for i in range(len(sequences))) + sum(Posterior_A[i] * len(sequences[i]) for i in range(len(sequences))))
        updated_P_B = (sum(Posterior_B[i] * sequences[i].count('H') for i in range(len(sequences))) + sum(Posterior_B[i] * sequences[i].count('T') for i in range(len(sequences))) ) / (sum(Posterior_B[i] * len(sequences[i]) for i in range(len(sequences))) + sum(Posterior_A[i] * len(sequences[i]) for i in range(len(sequences))))

        """        
        updated_P_A = sum(Posterior_A) / len(sequences)
        updated_P_B = sum(Posterior_B) / len(sequences)

        # Normalize Prior Probabilities to Ensure P_A + P_B = 1
        
        total_prior = updated_P_A + updated_P_B
        updated_P_A /= total_prior
        updated_P_B /= total_prior
        
        """
        
        print(f"Updated P_A: {updated_P_A:.4f}, P_B: {updated_P_B:.4f}")

        
        print(f"Updated Q_A: {updated_Q_A:.4f}, Q_B: {updated_Q_B:.4f}\n")
        
        
        # Convergence check
        if abs(updated_Q_A - Q_A) < tol and abs(updated_Q_B - Q_B) < tol and abs(updated_P_A - P_A) < tol:
            print("Convergence reached.")
            break

        # Update parameters
        Q_A, Q_B, P_A, P_B = updated_Q_A, updated_Q_B, updated_P_A, updated_P_B

    return Q_A, Q_B, P_A, P_B, Posterior_A, Posterior_B

# Generate data
data = generate_coin_data(10)
sequences = [sequence for _, sequence in data]

# Run EM Algorithm
Q_A, Q_B, P_A, P_B, Posterior_A, Posterior_B = em_algorithm(sequences)

# Print final results
print("\nFinal Parameters:")
print(f"Q_A (Head Probability for Coin A): {Q_A:.4f}")
print(f"Q_B (Head Probability for Coin B): {Q_B:.4f}")
print(f"P_A (Prior Probability for Coin A): {P_A:.4f}")
print(f"P_B (Prior Probability for Coin B): {P_B:.4f}\n")

# Assign each sequence to the most likely coin
assignments = []
for i, sequence in enumerate(sequences):
    if Posterior_A[i] > Posterior_B[i]:
        assignments.append(('A', sequence))
    else:
        assignments.append(('B', sequence))

# Print assignments
for coin, sequence in assignments:
    print(f"Sequence: {sequence} is likely to be Coin {coin}")
