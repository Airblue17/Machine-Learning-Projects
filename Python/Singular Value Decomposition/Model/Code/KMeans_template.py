import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# utility functions
def plot_mnist(elts, m, n):
    """Plot MNIST images in an m by n table. Note that we crop the images
    so that they appear reasonably close together.  Note that we are
    passed raw MNIST data and it is reshaped.

    Example: plot_mnist(X_train, 10, 10)
    """
    fig = plt.figure()
    images = [elt.reshape(28, 28) for elt in elts]
    img = np.concatenate([np.concatenate([images[m*y+x] for x in range(m)], axis=1)
                          for y in range(n)], axis=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img, cmap = matplotlib.cm.binary)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_line(X,Y,title,xlabel,ylabel):
    # X,Y,title,xlabel,ylabel
    # function to make line plot for given X,Y
    # plt.plot(X, Y, style)
    plt.plot(X, Y, 'b-')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True)
    plt.show()


class KMeans:
    """K-Means clustering with random centroid initialization.

    Arguments:
        K : (int) default=10
            number of centroids
    """

    def __init__(self, K=10):
        self.K = K
        self.centroids = None

    def _init_centroids(self, X):
        """Compute the initial centroids
        
        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.

            K : (int) 
                number of centroids

        Returns:
            centers : (ndarray, shape = (K, n_features))
        """
        K = self.K
        centers = np.zeros((self.K, X.shape[1]))
        #######################################
        # TODO: Write your code to randomly 
        # initialize K examples as centroids
        n_sample = X.shape[0]
        for i in range(K):
            k = np.random.choice(n_sample)
            centers[i,:] = X[k,:]
       

        #######################################
        return centers


    def _find_closest_centroids(self, X, centroids=None):
        """Finds the closest centroids for each example

        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.

            centroids : (None or ndarray, shape = (K, n_features))
                Cluster centers

        Returns:
            idx : (ndarray, shape = (n_samples,))
                Cluster id assigned to each example
        """
        if centroids is None:
            centroids = self.centroids

        distances = np.zeros((X.shape[0], centroids.shape[0]))
        idx = np.zeros((X.shape[0],))
        #######################################
        # TODO: Write your code to assign a 
        # cluster ID to each example
        distances = np.linalg.norm(X[:,None] - centroids, axis = 2)
        idx = np.argmin(distances, axis = 1)
        
        #######################################

        return idx

    def _compute_centroids(self, X, idx):
        """Returns the new centroids by computing the means of the 
        data points assigned to each centroid

        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.

            idx : (ndarray, shape = (n_samples,))
                Cluster ids for each example

            K : (int) 
                number of centroids

        Returns:
            centroids : (ndarray, shape = (K, n_features))
                new cluster centers formed by taking means of data points
                assigned to old centroids
        """
        K = self.K
        centroids = np.zeros((self.K, X.shape[1]))
        #######################################
        # TODO: Write your code to compute the 
        # new centroids by computing the means of 
        # data points assigned to each cluster
        # Try to use maximum 1 loop over number of
        # samples in X
        for i in range(K):
            centroids[i,:] = np.mean(X[idx==i], axis = 0)

        #######################################
        return centroids

    def score(self, X, idx=None, centroids=None):
        """Compute the score function

        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.

            idx : (None or ndarray, shape = (n_samples,))
                Cluster ids for each example

            centroids : (None or ndarray, shape = (K, n_features))
                Cluster centers

        Returns:
            score : (float) 
                negative within cluster scattering
        """
        if idx is None and centroids is None:
            idx, centroids = self.predict(X), self.centroids
        score = 0.0
        for k in range(self.K):
            score -= np.sum(np.linalg.norm(X[idx==k] - centroids[k], axis=1))
        return score

    def fit(self, X, max_iters=100, verbose=False):
        """Compute K-means clustering

        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.
            
            max_iters : (int)
                Maximum number of iteration to run K-Means algorithm

            verbose : (bool), default=False
                Verbosity mode 
        """
        prev_max_diff = 0.0
        centroids = self._init_centroids(X)

        for i in range(max_iters):

            idx = self._find_closest_centroids(X, centroids)
            prev_centroids = centroids
            centroids = self._compute_centroids(X, idx)

            if verbose:
                score = self.score(X, idx, centroids)
                print ("Running K-Means Iteration : {}, Score : {}".format(i+1, score))

            # early stopping
            diffs = map(lambda u, v: np.linalg.norm(u-v),prev_centroids,centroids)
            max_diff = max(diffs)
            diff_change = abs((max_diff-prev_max_diff)/np.mean([prev_max_diff,max_diff])) * 100
            prev_max_diff = max_diff

            if np.isnan(diff_change) or diff_change < 1e-4:
                print ("Converged!")
                break

        # Store the final centroids
        self.centroids = centroids

    def predict(self, X):
        """Predicts cluster ids for samples in X based on current centroids

        Arguments:
            X : (ndarray, shape = (n_sample, n_features))
                Training input matrix where each row is a feature vector.

        Returns:
            idx : (ndarray, shape = (n_samples,))
                Cluster id assigned to each example
        """
        idx = np.zeros((X.shape[0]),)
        #######################################
        # TODO: Write your code to predict the 
        # cluster IDs for a given set of examples
        idx = self._find_closest_centroids(X)
        

        #######################################
        return idx


def main():
    # Read the data
    X_train = np.load('../../Data/X_train.npy')
    X_val = np.load('../../Data/X_val.npy')
    best_score = -99999999
    best_km = KMeans(K=10)

    print("Question 3.1")
    for i in range(10):
        print("Run ", i+1)
        np.random.seed()
        km = KMeans(K=10)
        km.fit(X_train, verbose=False)
        score = km.score(X_val)
        print ("score = {}".format(score))
        if(score > best_score):
            best_km.centroids = km.centroids
            best_score = score
    idx = best_km.predict(X_val)
    K = best_km.K
    for i in range(K):
        print("Cluster ", i+1)
        X_val_clus = X_val[idx==i]
        plot_mnist(X_val_clus, 5, 5)
        
    print("\nQuestion 3.2")  
    np.random.seed(32)
    K_val = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    score_k = []
    for i in range(len(K_val)):
        print("For K = ", K_val[i])
        km = KMeans(K=K_val[i])
        km.fit(X_train, verbose=False)
        score = km.score(X_val)
        print("Score Calculated\n")
        score_k.append(score)
    plot_line(K_val,score_k,"(Q3.2) Score on Validation set vs K","K","Score on Validation set")

if __name__ == '__main__':
    main()
