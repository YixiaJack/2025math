import pandas as pd
import matplotlib.pyplot as plt
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "College.csv")

# (a)
college = pd.read_csv(csv_path)
#(b)
college3 = college.rename({'Unnamed: 0': 'College'}, axis=1)
college3 = college3.set_index('College')
college3.head()
college = college3
#(c)
college.describe()
#(d)
pd.plotting.scatter_matrix(college[['Top10perc', 'Apps', 'Enroll']], 
                           figsize=(10, 10), diagonal='hist')
plt.suptitle("Scatterplot Matrix for Top10perc, Apps, Enroll", fontsize=14)
plt.show()
#(e)
college.boxplot(column='Outstate', by='Private', figsize=(8, 6))
plt.title("Out-of-State Tuition by Private/Public")
plt.suptitle("") 
plt.xlabel("Type of Institution (Private = Yes/No)")
plt.ylabel("Out-of-State Tuition")
plt.show()
#(f)
college['Elite'] = pd.cut(college['Top10perc'], [0, 50, 100], labels=['No', 'Yes'])
print(college['Elite'].value_counts())
college.boxplot(column='Outstate', by='Elite', figsize=(8, 6))
plt.title("Out-of-State Tuition by Elite Status")
plt.suptitle("")
plt.xlabel("Elite")
plt.ylabel("Out-of-State Tuition")
plt.show()
#(g)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].hist(college['Apps'], bins=20, color='skyblue', edgecolor='black')
axs[0, 0].set_title('Histogram of Applications')
axs[0, 0].set_xlabel('Applications')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(college['Accept'], bins=20, color='salmon', edgecolor='black')
axs[0, 1].set_title('Histogram of Acceptances')
axs[0, 1].set_xlabel('Acceptances')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(college['Enroll'], bins=20, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('Histogram of Enrollments')
axs[1, 0].set_xlabel('Enrollment')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(college['Grad.Rate'], bins=20, color='plum', edgecolor='black')
axs[1, 1].set_title('Histogram of Graduation Rates')
axs[1, 1].set_xlabel('Graduation Rate')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
#(h)
#Based on the exploratory data analysis of the College dataset, several key patterns emerge:
#Tuition Disparities: Private institutions consistently charge higher out-of-state tuition compared to public universities, as evidenced by the boxplot comparisons.
#Elite Institution Characteristics: The analysis reveals that only a select portion of schools have more than 50% of their students coming from the top 10% of their high school classes (defined as "Elite" schools). These Elite institutions typically command higher out-of-state tuition rates.
#Admissions Relationships: The scatterplot matrix examining Top10perc (percentage of students from top 10% of high school class), Apps (number of applications), and Enroll (enrollment figures) suggests potential correlations between these variables. Schools receiving more applications tend to have higher enrollment numbers, though the relationship shows clustering and outliers.
#Variable Distributions: The histograms for applications, acceptances, enrollment, and graduation rates display varying distribution shapes, with some variables (particularly Apps and Accept) showing right-skewed distributions, indicating that a small number of institutions have exceptionally high values compared to the majority.
#These preliminary findings provide a foundation for more sophisticated analyses to understand factors influencing college admissions processes and the distinguishing features of elite educational institutions.