import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def SVD(A, s, k):
    np.random.seed(10)
    # TODO: Calculate probabilities p_i
    n,m = A.shape
    p_i = np.zeros((n,))
    denom = np.linalg.norm(A, ord ='fro')**2
    for i in range(n):
        p_i[i] =  np.linalg.norm(A[i,:])**2 / denom
        
    # TODO: Construct S matrix of size s by m
    S = np.zeros((s,m))
    
    j_idx = np.random.choice(n, size=s, p=p_i)
    S = A[j_idx, :].astype(float)
    
    # TODO: Calculate SS^T
    SST = S.dot(S.T)
    
    
    # TODO: Compute SVD for SS^T
    u, s, vh = np.linalg.svd(SST)
        
    # TODO: Construct H matrix of size m by k
    H = np.zeros((m,k))
    
    for t in range(k):
        H[:,t] = S.T.dot(u[:,t]) / np.linalg.norm( S.T.dot(u[:,t]), ord = 2)
  

    # Return matrix H and top-k singular values sigma
    s = -np.sort(-s) 
    
    return H, s[:k] # H and top-k singular values
def main():
    im = Image.open("../../Data/baboon.tiff")
    A = np.array(im)
    H, sigma = SVD(A, 80, 60)
    k = 60
    #A = np.arange(1,17).reshape(4,4)
    #H, sigma = SVD(A, 3, 2)
    #k = 2

    # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.
    u, sigma_ok, v = np.linalg.svd(A)

    sigma_ok2 = -np.sort(-sigma_ok)
    sigma_ok2 = np.diag(sigma_ok2[:k])
    
    u_ok = u[:,:k]
    v_ok = v[:k,:]
    recon_ok = u_ok.dot(sigma_ok2.dot(v_ok))
    
  
    
    # TO DO: Use H to compute sub-optimal k rank approximation for A
    
    v_sok = H.T #60x512
    sigma_sok = sigma
    sigma_sok = np.diag(sigma_sok) #60x60
    u_sok = A.dot(v_sok.T).dot(np.linalg.inv(sigma_sok))

    recon_sok = u_sok.dot(sigma_sok.dot(v_sok))
    # To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation
    plt.figure(figsize = (9,9))
    plt.subplot(1,3,1)
    plt.imshow(A, cmap = "gray")
    plt.title("Orginal")
    plt.axis("off")
    
    
    plt.subplot(1,3,2)
    plt.imshow(recon_ok, cmap = "gray")
    plt.title("Optimal k-rank")
    plt.axis("off")
    
    
    plt.subplot(1,3,3)
    plt.imshow(recon_sok, cmap = "gray")
    plt.title("Sub-Optimal k-rank")
    plt.axis("off")
    
    plt.savefig("../Figures/svdPlot.png")
    
    # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    # sub-optimal k-rank approximation for A using H.
    diff_ok = A - recon_ok
    diff_sok = A - recon_sok
    
    err_ok =  np.linalg.norm(diff_ok, ord ='fro')
    err_sok =  np.linalg.norm(diff_sok, ord ='fro')
    
    print("Error in terms of the frobenius norm")
    print("For optimal k-rank:", err_ok)
    print("For sub-optimal k-rank:", err_sok)
    
if __name__ == "__main__":
     main()
