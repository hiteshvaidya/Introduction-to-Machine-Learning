"""
CSCI 635: Home work 1
Name: Hitesh Ulhas Vaidya
email: hv8322@rit.edu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Visualize(object):

    def __init__(self):
        self.file1 = np.array([])
        self.file2 = np.array([])

    def readFile(self):
        self.file1 = pd.read_csv('Frogs.csv', header=0, index_col=False)
        self.file2 = pd.read_csv('Frogs-subsample.csv', header=0,
                                 index_col=False)

    def helper_class_split(self, file_name):
        f1 = list(file_name)[0]
        f2 = list(file_name)[1]
        data = file_name.to_numpy()
        class1 = data[data[:, 2] == 'HylaMinuta']
        class2 = data[data[:, 2] == 'HypsiboasCinerascens']
        return f1, f2, class1, class2

    def plot_scatter(self, file_name, name):
        f1, f2, class1, class2 = self.helper_class_split(file_name)

        plt.scatter(class1[:,0], class1[:,1], c='red',
                    label="HylaMinuta")
        plt.scatter(class2[:,0], class2[:,1], c='blue',
                    label="HypsiboasCinerascens")
        plt.title('Scatter plot for both classes in ' + name)
        plt.legend(loc='upper right')
        plt.xlabel(f1 + ' values')
        plt.ylabel(f2 + ' values')
        plt.show()

        # plt.scatter(np.arange(class1.shape[0]), class1[:, 1], c='red',
        #             label="HylaMinuta")
        # plt.scatter(np.arange(class2.shape[0]), class2[:, 1], c='blue',
        #             label="HypsiboasCinerascens")
        # plt.title('Plotting feature ' + f2 + ' for both classes in ' + name)
        # plt.legend(loc='upper right')
        # plt.xlabel("Number of data points" "HylaMinuta", )
        # plt.ylabel(f2 + " values")
        # plt.show()

    def plot_histogram(self, file_name, name):
        f1, f2, class1, class2 = self.helper_class_split(file_name)

        plt.hist((class1[:,0],class1[:,1]), color=('orange','green'), bins=20,
        label=(f1, f2))
        plt.title('histograms of featues for class HylaMinuta from ' + name)
        plt.xlabel('Number of data points')
        plt.ylabel('Values of features')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist((class2[:, 0], class2[:, 1]), color=('blue', 'red'),
                 bins=20, label=(f1, f2))
        plt.title('histograms of featues for class HypsiboasCinerascens from '
                  '' + name)
        plt.xlabel('Number of datapoints')
        plt.ylabel('Values of features')
        plt.legend(loc='upper right')
        plt.show()

    def plot_line_graph(self, file_name, name):
        f1, f2, class1, class2 = self.helper_class_split(file_name)

        t1 = np.sort(class1[:,0])
        t2 = np.sort(class1[:,1])
        plt.plot(np.arange(t1.shape[0]), t1, color='orange', label=f1)
        plt.plot(np.arange(t2.shape[0]), t2, color='green', label=f2)
        plt.title('Line graph of features for HylaMinuta from ' + name)
        plt.xlabel('Number of data points')
        plt.ylabel('Values of features')
        plt.legend(loc='upper right')
        plt.show()

        t1 = np.sort(class2[:, 0])
        t2 = np.sort(class2[:, 1])
        plt.plot(np.arange(t1.shape[0]), t1, color='orange', label=f1)
        plt.plot(np.arange(t2.shape[0]), t2, color='green', label=f2)
        plt.title('Line graph of features for HypsiboasCinerascens from ' + name)
        plt.xlabel('Number of data points')
        plt.ylabel('Values of features')
        plt.legend(loc='upper right')
        plt.show()

    def plot_boxplot(self, file_name, name):
        f1, f2, class1, class2 = self.helper_class_split(file_name)

        plt.boxplot(class1[:,:2])
        plt.title('Boxplot for class HylaMinuta from ' + name)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

        plt.boxplot(class2[:, :2])
        plt.title('Boxplot for class HypsiboasCinerascens from ' + name)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

    def plot_errorbar(self, file_name, name):
        f1, f2, class1, class2 = self.helper_class_split(file_name)

        x_pos = np.arange(4)
        CTEs = [np.mean(class1[:,0]), np.mean(class1[:,1]),
                np.mean(class2[:,0]), np.mean(class2[:,1])]
        error = [np.std(class1[:,0]), np.std(class1[:,1]),
                np.std(class2[:,0]), np.std(class2[:,1])]
        plt.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,
                ecolor='black', capsize=10)
        plt.tight_layout()
        plt.title('Bar graph with error bars for ' + name)
        plt.show()

    def get_stats(self, file_name, name):
        f1 = list(file_name)[0]
        f2 = list(file_name)[1]
        data = file_name.to_numpy()

        f1_mean = np.mean(data[:,0])
        f2_mean = np.mean(data[:,1])
        f1_std = np.std(data[:,0])
        f2_std = np.std(data[:,1])

        temp = data[:,:2].astype(float)
        cov_matrix = np.cov(temp[:,:2].T)
        print('Statistics for',name)
        print('mean of', f1, ': ', f1_mean)
        print('mean of', f2, ': ', f2_mean)
        print('standard deviation of', f1, ': ', f1_std)
        print('standard deviation of', f2, ': ', f2_std)
        print('covariance matrix:\n', cov_matrix)
        print(
            '\n-------------------------------------------------------------\n')


    def run(self):
        self.readFile()

        self.plot_scatter(self.file1, 'Frogs.csv')
        self.plot_scatter(self.file2, 'Frogs-subsample.csv')
        print('Scatter plot done')

        self.plot_histogram(self.file1, 'Frogs.csv')
        self.plot_histogram(self.file2, 'Frogs-subsample.csv')
        print('histogram plot done')

        self.plot_line_graph(self.file1, 'Frogs.csv')
        self.plot_line_graph(self.file2, 'Frogs-subsample.csv')
        print('Line graph done')

        self.plot_boxplot(self.file1, 'Frogs.csv')
        self.plot_boxplot(self.file1, 'Frogs-subsample.csv')
        print('boxplot done')

        self.plot_errorbar(self.file1, 'Frogs.csv')
        self.plot_errorbar(self.file2, 'Frogs-subsample.csv')
        print('Bar graph with errors done')

        self.get_stats(self.file1, 'Frogs.csv')
        self.get_stats(self.file2, 'Frogs-subsample.csv')


if __name__ == '__main__':
    vis = Visualize()
    vis.run()