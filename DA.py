import sklearn as skl
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T = None
        self.T_min = None

        self.cluster_centers = None
        self.cluster_probs = None

        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()

        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """
        # TODO:
        # (2) Initialize
        K = 1
        alpha = 0.95
        N = samples.shape[0]
        d = samples.shape[1]
        y1 = np.sum(samples, axis=0)/N #center of mass
        self.cluster_centers = np.array([y1]) #the first cluster center is the center of mass of all samples
        self.cluster_probs = np.array([1]) #p(y1) = 1 because K=1
        #initial temperature
        C_x = np.zeros((d,d))
        for i in range(N):
            C_x += (samples[i,:]-y1) @ (samples[i,:]-y1).T
        C_x /= N
        w, _ = np.linalg.eig(C_x) #compute eigenvalues of C_x
        self.T = 2.5*np.max(w) #T>2*lambda_max(C_x)
        self.T_min = 0.1 # what should this be?
        not_converged = True


        while(self.T >= self.T_min): # (5) while T>Tmin
            while(not_converged): # (4) while not converged

                #compute probabilities
                distance = self.get_distance(samples, self.cluster_centers)
                prob = self._calculate_cluster_probs(distance, self.T)
                self.cluster_probs = np.sum(prob, axis=0)/N
                old_y = np.copy(self.cluster_centers)

                for i in range(K): # (3) update each cluster center
                    self.cluster_centers[i,:] = (np.dot(samples.T,prob[:,i]))/(N*self.cluster_probs[i])

                #check if converged
                not_converged = (np.linalg.norm(old_y-self.cluster_centers, axis=1)>0.1).sum() #how to check for convergence???

            # (6) cooling step t = t*alpha (apha<1)
            self.T = alpha*self.T
            print(self.T)
            print(K)

            for i in range(K):
                distance = self.get_distance(samples, self.cluster_centers)
                prob = self._calculate_cluster_probs(distance, self.T)
                self.cluster_probs = np.sum(prob, axis=0)/N
                if(K < self.n_clusters): # (7)
                    # compute critical temperature
                    C_x = np.zeros((d,d))
                    for j in range(N):
                        C_x += prob[j,i] * (samples[j,:]-self.cluster_centers[i]) @ ((samples[j,:]-self.cluster_centers[i]).T)
                    C_x /= N*self.cluster_probs[i]
                    w, _ = np.linalg.eig(C_x) #compute eigenvalues of C_x
                    T_crit = 2*np.max(w)
                    if(self.T <= T_crit): # (8) if critical temp is reached
                        # add a new codevector
                        y_new = self.cluster_centers[i,:]+np.random.uniform(1,4,samples.shape[1])
                        self.cluster_centers = np.append(self.cluster_centers, y_new.reshape(-1,samples.shape[1]), axis=0)
                        #update cluster probabilities
                        p_new = self.cluster_probs[i]/2
                        self.cluster_probs[i] /= 2
                        self.cluster_probs = np.append(self.cluster_probs, p_new)
                        K+=1














    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        # TODO:
        probs = np.exp(-dist_mat/temperature)
        probs /= np.sum(probs, axis=1)[:,None]

        return probs

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        # TODO:
        n_samples = samples.shape[0]
        n_cengtroids = clusters.shape[0]
        D = np.zeros((n_samples, n_cengtroids))

        for j in range(n_cengtroids):
            D[:,j] = np.linalg.norm(samples-clusters[j,:],axis=1)**2 #maybe: metric = self.metric ???

        return D

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        for node in self.bifurcation_tree.all_nodes_itr():
            c_id = node.data['cluster_id']
            my_dist = node.data['distance']

            if c_id > 0 and len(clusters[c_id]) == 0:
                clusters[c_id] = list(np.copy(clusters[c_id-1]))
            clusters[c_id].append(my_dist)

        # Cut the last iterations, usually it takes too long
        cut_idx = self.bifurcation_tree_cut_idx + 20

        beta = [1 / t for t in self.temperatures]
        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):
            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
