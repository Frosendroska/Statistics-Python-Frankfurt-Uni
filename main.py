import numpy as np
from matplotlib import pyplot as plt

# (b) load data from body data into numpy array
data = np.genfromtxt("Body-Data.csv", delimiter=",", skip_header=1)
# (c) Sort individual columns of matrix
# (d) Sort the rows of the original matrix by age
data = data[data[:, 1].argsort()]
# (e) median
print("Median of age = ", np.median(data[:, 1]))
print("Median of height = ", np.median(data[:, 2]))
print("Median of weight = ", np.median(data[:, 3]), "\n")
# (e) Mean
print("Mean of age = ", np.nanmean(data[:, 1]))
print("Mean of height = ", np.nanmean(data[:, 2]))
print("Mean of weight = ", np.nanmean(data[:, 3]), "\n")
# (e) Variance
print("Variance of age = ", np.nanvar(data[:, 1]))
print("Variance of height = ", np.nanvar(data[:, 2]))
print("Variance of weight = ", np.nanvar(data[:, 3]), "\n")
# (e) Standard deviation
print("Variance of age = ", np.nanstd(data[:, 1]))
print("Variance of height = ", np.nanstd(data[:, 2]))
print("Variance of weight = ", np.nanstd(data[:, 3]), "\n")

# (f) Covariance and the correlation matrix
x = data[:, 2]  # height
y = data[:, 3]  # weight
cov_data = np.cov(x, y)
print("Covariance matrix: \n", np.cov(cov_data))
print("Сorrelation matrix: \n", np.corrcoef(cov_data))

# (g) Plot the absolute frequency distributions and histograms
plt.hist(data[:, 1])
plt.title("absolute frequency distribution (age)")
plt.show()
plt.hist(data[:, 1], density=True)
plt.title("histogram (age)")
plt.show()

plt.hist(data[:, 2])
plt.title("absolute frequency distribution (height)")
plt.show()
plt.hist(data[:, 2], density=True)
plt.title("histogram (height)")
plt.show()

plt.hist(data[:, 3])
plt.title("absolute frequency distribution (weight)")
plt.show()
plt.hist(data[:, 3], density=True)
plt.title("histogram (weight)")
plt.show()

# (h) Create scatterplots for every combination of variables
a = data[:, 1]
b = data[:, 2]
plt.scatter(a, b)
plt.title("scatterplot (age, height)")
plt.show()

a = data[:, 2]
b = data[:, 3]
plt.scatter(a, b)
plt.title("scatterplot (weight, height)")
plt.show()

a = data[:, 1]
b = data[:, 3]
plt.scatter(a, b)
plt.title("scatterplot (weight, age)")
plt.show()

# (i) Perform a regression analysis with weight as dependent and height as independent variable.
#      Compute a, b, and y^
x = data[:, 2]  # indep
y = data[:, 3]  # dep
b = np.cov(x, y) / np.nanvar(x)
a = np.nanmean(y) - b * np.nanmean(x)
y_ = []
for i in x:
    y_.append(a + b * i)
y_hat = np.array(y)
plt.scatter(x, y)
plt.plot(x, y_hat, lw=4, c='orange')
plt.title("Regression analysis")
plt.show()

