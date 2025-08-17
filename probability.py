import numpy as np


def poisson_probability(k, lam): 
    a = lam ** k
    b = np.e**(-lam)
    c = np.math.factorial(k)
    p_k = a*b/c
    return round(p_k,5) 

def poisson_distribution(lam, k):
    return np.exp(-lam) * (lam ** k) / np.math.factorial(k)

if __name__ == "__main__":
    k = int(input("Enter the number of events: "))
    lam = float(input("Enter the average rate: "))
    print(poisson_probability(k, lam))