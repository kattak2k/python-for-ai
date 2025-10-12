
# --- Import llm response functions from helper_functions.py ---
# do not use
# from helper_functions import *
from helper_functions import get_llm_response, print_llm_response

# Import from the math package the cos, sin and pi functions
from math import cos, sin, tan, pi, floor
from math import sqrt as square_root  # Import sqrt function with an alias
print(pi)

values = [0, pi/2, pi, 3/2*pi, 2*pi]
for value in values:
    print(f"The cosine of {value:.2f} is {cos(value):.2f}")
    print(f"The sine of {value:.2f} is {sin(value):.2f}")
    print(f"The tangent of {value:.2f} is {tan(value):.2f}")
    
floor(5.7)
square_root(25)


# Statistics
from statistics import mean, stdev, median, mode, variance
my_friends_heights = [160, 172, 155, 180, 165, 170, 158, 182, 175, 168]
# mean value of the list of heights
mean(my_friends_heights)
#calculate the standard deviation
stdev(my_friends_heights)
# Calculate the median score
median_score = median(my_friends_heights)
print(median_score)

# random
from random import random, randint, choice
random_spices = sample(spices, 2)
random_vegetables = sample(vegetables, 2)
random_protein = sample(proteins, 1)
print(random_protein)
