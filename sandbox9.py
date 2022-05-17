
A = np.array([[vi**i for i in range(4)] for vi in [-1.9027532339607447, 1.7606909524780678, 2.2640558571368445,  2.6815879179748796]])
coefficients = inv(A.transpose() @ A) @ A.transpose()) @ y
y = np.array([-1, 1, -1, 1])

# Evaluate the polynomial in terms of the new basis
((inv(A.transpose() @ A) @ A.transpose()) * y).transpose() @ np.array([z**0, z**1, z**2, z**3])
