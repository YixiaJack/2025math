#Quantitative Variables (numeric):
#mpg: Miles per gallon (fuel efficiency; usually the response when predicting fuel economy)
#cylinders: Number of cylinders
#displacement: Engine displacement
#horsepower: Engine horsepower
#weight: Vehicle weight
#acceleration: Time to accelerate from 0 to 60 mph (or similar measure)
#year: Model year (treated as numeric for most analyses)
#Qualitative Variables:
#origin: Coded as 1 (American), 2 (European), and 3 (Japanese). Although stored as numbers, it represents categories.
#name: The name of the vehicle (text/string)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Auto.data")

auto = pd.read_csv(data_path, na_values='?', sep='\s+').dropna()

print("DataFrame columns:", auto.columns.tolist())
print("First few rows:")
print(auto.head())

# (b) Compute the Range (Minimum and Maximum) for each Quantitative Predictor
quant_vars = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year"]

print("\nRange (min and max) for each quantitative predictor:")
for var in quant_vars:
    auto[var] = pd.to_numeric(auto[var], errors='coerce')
    min_val = auto[var].min()
    max_val = auto[var].max()
    print(f"{var:15s}: min = {min_val}, max = {max_val}")

# (c) Mean and Standard Deviation
print("\nMean and Standard Deviation for each quantitative predictor:")
for var in quant_vars:
    mean_val = auto[var].mean()
    std_val = auto[var].std()
    print(f"{var:15s}: mean = {mean_val:.2f}, std = {std_val:.2f}")

# (d) Remove the 10th through 85th observations
auto_subset = pd.concat([auto.iloc[:9], auto.iloc[85:]])
print(f"\nShape after removing 10th through 85th observations: {auto_subset.shape}")

# (e) Scatterplot matrix
pd.plotting.scatter_matrix(auto[quant_vars], figsize=(12, 12), diagonal='hist')
plt.suptitle("Scatterplot Matrix for Quantitative Variables", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# (f) Correlation matrix
corr_matrix = auto[quant_vars].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# (g) MPG vs. Horsepower scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(auto['horsepower'], auto['mpg'], alpha=0.7, edgecolor='k')
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("MPG vs. Horsepower")
plt.show()
'''
(b) Range of Each Quantitative Predictor
mpg: minimum = 9.0, maximum = 46.6
cylinders: minimum = 3, maximum = 8
displacement: minimum = 68.0, maximum = 455.0
horsepower: minimum = 46.0, maximum = 230.0
weight: minimum = 1613.0, maximum = 5140.0
acceleration: minimum = 8.0, maximum = 24.8
year: minimum = 70, maximum = 82
(Calculated using the .min() and .max() methods.)

(c) Mean and Standard Deviation of Each Quantitative Predictor
mpg: mean ≈ 23.45, standard deviation ≈ 7.81
cylinders: mean ≈ 5.47, standard deviation ≈ 1.71
displacement: mean ≈ 194.41, standard deviation ≈ 104.64
horsepower: mean ≈ 104.47, standard deviation ≈ 38.49
weight: mean ≈ 2977.58, standard deviation ≈ 849.40
acceleration: mean ≈ 15.54, standard deviation ≈ 2.76
year: mean ≈ 75.98, standard deviation ≈ 3.68
(Computed using the .mean() and .std() methods.)

(d) Subset After Removing the 10th through 85th Observations
Shape of the subset: (316, 9)
Range, Mean, and Standard Deviation:
The statistical summaries (range, mean, and standard deviation) for each quantitative predictor are slightly adjusted compared to the full dataset, but the overall distribution patterns remain very similar. The removal does not drastically change the relationships among variables.
(e) Graphical Investigation of the Predictors
Scatterplot Matrix:
A scatterplot matrix of the quantitative predictors reveals:
A strong negative relationship between mpg and variables such as weight, horsepower, and displacement.
High positive correlations among cylinders, displacement, horsepower, and weight.
A modest positive trend between year and mpg, indicating that newer cars tend to have better fuel efficiency.
Individual Scatterplots (Examples):
MPG vs. Weight: Demonstrates that as weight increases, mpg decreases sharply.
MPG vs. Horsepower: Shows that higher horsepower is associated with lower mpg.
(f) Predicting Gas Mileage (mpg)
Findings:
Weight exhibits a strong negative linear relationship with mpg—a key predictor.
Horsepower and displacement also show clear negative correlations with mpg.
Cylinders, due to their relationship with engine size, further support predicting mpg.
Year indicates that newer cars tend to be more fuel efficient.
Conclusion:
Yes, the plots suggest that weight, horsepower, and displacement (along with cylinders and possibly year) are useful predictors for mpg. These variables consistently exhibit strong inverse relationships with gas mileage, making them excellent candidates for building a predictive model.'''