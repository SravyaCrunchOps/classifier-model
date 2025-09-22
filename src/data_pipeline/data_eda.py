import seaborn as sns
import matplotlib.pyplot as plt


def eda(df):
    print(df.describe())

    class_type_counts = df['class_type'].value_counts()
    print(f'No.of class type: {class_type_counts}')

    # Visualizing class_type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='class_type', data=df, palette='viridis')
    plt.title('Distribution of Class Types')
    plt.xlabel('Class Type')
    plt.ylabel('Count')
    plt.show()
