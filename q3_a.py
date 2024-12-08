import random
import math
from math import comb


def binomial_probability(heads, trials, prob_head):
    """
    Binom dağılımı ile bir dizideki 'Head' gelme olasılığını hesaplar.
    
    heads: Gerçekleşen 'H' sayısı
    trials: Toplam deneme sayısı (örneğin, 10 atış)
    prob_head: Her bir atışta 'H' gelme olasılığı (örneğin, 0.5)
    """
    probability = comb(trials, heads) * (prob_head ** heads) * ((1 - prob_head) ** (trials - heads))
    return probability

def generate_coin_data(num_sequences=10):
    data = []
    for _ in range(num_sequences):
        coin_type = random.choice(['A', 'B'])  # Rastgele coin seç
        sequence = ''.join(random.choice(['H', 'T']) for _ in range(10))  # 10 atış üret
        data.append((coin_type, sequence))
    return data

def count_heads_and_tails(data):
    results = []
    for coin, sequence in data:
        heads_count = sequence.count('H')  # Dizideki H sayısını bul
        tails_count = sequence.count('T')  # Dizideki T sayısını bul
        results.append({
            'coin': coin,
            'sequence': sequence,
            'heads_count': heads_count,
            'tails_count': tails_count
        })
    return results

# Örnek kullanım
data = generate_coin_data(5)  # 5 set data üret
results = count_heads_and_tails(data)

# Sonuçları yazdır
for result in results:
    print(f"Coin: {result['coin']}, Sequence: {result['sequence']}, Heads: {result['heads_count']}, Tails: {result['tails_count']}")

# Sayılara nasıl ulaşılır?
# Örneğin, ilk dizi için
first_result = results[0]
print("\nFirst sequence details:")
print(f"Coin: {first_result['coin']}, Heads: {first_result['heads_count']}, Tails: {first_result['tails_count']}")


# İlk dizinin head (H) sayısını almak için:
first_heads_count = results[0]['heads_count']
print(f"First sequence's head count: {first_heads_count}\n")

# Tüm head (H) sayılarını bir listeye toplamak için:
all_heads_counts = [result['heads_count'] for result in results]
print(f"All heads counts: {all_heads_counts}")

# Her bir sonuç için olasılığı hesapla
for result in results:
    prob = binomial_probability(result['heads_count'], trials=10, prob_head=0.5)
    print(f"Coin: {result['coin']}, Sequence: {result['sequence']}, Probability of Head Count: {prob:.4f}\n")
    

def em_algorithm(sequences, max_iter=100, tol=1e-6):
    # Başlangıç parametreleri
    Q_A = 0.6  # Coin A'nın Head gelme olasılığı
    Q_B = 0.5  # Coin B'nin Head gelme olasılığı
    P_A = 0.3  # Prior probability of choosing Coin A
    P_B = 0.7  # Prior probability of choosing Coin B
    for iteration in range(max_iter):
        # E Adımı: Coin A ve Coin B'ye ait olma olasılıklarını hesapla
        Q_A_values = []
        Q_B_values = []

        for sequence in sequences:
            heads = sequence.count('H')
            tails = sequence.count('T')
            trials = heads + tails

            # Likelihood hesapla (binom katsayıları eklenmiş)
            likelihood_A = (comb(trials, heads) * (Q_A ** heads) * ((1 - Q_A) ** tails) ) * P_A
            likelihood_B = ( comb(trials, heads) * (Q_B ** heads) * ((1 - Q_B) ** tails) ) * P_B

            # Normalize ederek Q değerlerini bul
            total_likelihood = likelihood_A + likelihood_B
            Q_A_values.append(likelihood_A / total_likelihood)
            Q_B_values.append(likelihood_B / total_likelihood)

        # M Adımı: Q_A ve Q_B'yi güncelle
        new_Q_A = sum(Q_A_values[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                  sum(Q_A_values[i] * len(sequences[i]) for i in range(len(sequences)))

        new_Q_B = sum(Q_B_values[i] * sequences[i].count('H') for i in range(len(sequences))) / \
                  sum(Q_B_values[i] * len(sequences[i]) for i in range(len(sequences)))

        # Durma kriteri
        if abs(new_Q_A - Q_A) < tol and abs(new_Q_B - Q_B) < tol:
            break

        # Güncellenen parametreleri ata
        Q_A = new_Q_A
        Q_B = new_Q_B

    return Q_A, Q_B, Q_A_values, Q_B_values


data = generate_coin_data(5)  # 5 set data üret
results = count_heads_and_tails(data)
sequences = [result['sequence'] for result in results ]

Q_A, Q_B, Q_A_values, Q_B_values = em_algorithm(sequences)

print(f"Final Q_A (Head Probability for Coin A): {Q_A:.4f}\n")
print(f"Final Q_B (Head Probability for Coin B): {Q_B:.4f}\n")
print(f"Q_A values (per sequence): {Q_A_values}\n")
print(f"Q_B values (per sequence): {Q_B_values}\n")


# EM algoritması sonrası hangi dizinin hangi coine ait olduğunu belirleme
assignments = []  # Dizilerin coine atanması için bir liste
for i, sequence in enumerate(sequences):
    if Q_A_values[i] > Q_B_values[i]:  # Q_A değeri büyükse Coin A'ya atanır
        assignments.append(('A', sequence))
    else:  # Q_B değeri büyükse Coin B'ye atanır
        assignments.append(('B', sequence))

# Sonuçları yazdır
for coin, sequence in assignments:
    print(f"Sequence: {sequence} is assigned to Coin {coin}")
