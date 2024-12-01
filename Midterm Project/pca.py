from sklearn.decomposition import PCA
from load_mnist import load

def pca_alg(x, num_dim=2):

    pca = PCA(n_components=num_dim)
    x_reduced = pca.fit_transform(x)
    return x_reduced

def main():
    x, _ = load()
    pca_alg(x)

if __name__ == '__main__':
    main()