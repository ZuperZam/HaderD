def MOON_GetDataSet(n_samples):
    X,y = make_moons(n_samples= n_samples, noise = 0.05)
    return X,y

def MOON_Plot(X, y):
     plt.scatter(X[:,0], X[:,1], s=40, c=y)
        

        
        
        
def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
from sklearn.datasets import fetch_mldata

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image)
    plt.axis("off")
    plt.show

def MNIST_GetDataSet():
    fetch_mnist()
    mnist = fetch_mldata('MNIST original')
    return(mnist["data"], mnist["target"])




def IRIS_GetDataSet():
    fetch_mnist()
    mnist = sklearn.datasets.load_iris()
    return(mnist["data"], mnist["target"])