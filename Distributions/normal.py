import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma):
    y = 1/(np.sqrt(sigma**2 * np.pi * 2)) * np.exp(-1/2 * ((x- mu)/sigma)**2)
    return y

# TODO: Feed random variable X instead of pdf into log_pdf() function
def log_pdf(norm_dist, mu, sigma):
    y = np.exp(mu + sigma * np.array(norm_dist)
               )
    return y
# test = normal_pdf(1, 0, 1)

y_values = []
num_points = 1001
mu = 0
sigma = 1
for x in np.linspace(start = -3, stop = 3, num = num_points):
    y_values.append(normal_pdf(x, mu, sigma))

# test = plt.plot(y_values)

test = log_pdf(y_values, 0, 1)

print('done')