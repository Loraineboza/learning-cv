from batch_norm import BN
import numpy as np 

if __name__ == "__main__":
    np.random.seed(1)
    
    B, D = 4, 2
    ep = 5
    lr = 3e-1

    X = np.random.randn(B, D) * 10.0 + 5.0 #mean=5; std=10
    true = np.array([[1.0, -1.0], [0.5, 0.5], [-0.5, 0.0], [2.0, -2.0]])
    
    bn = BN(num_features=D)

    for e in range(1, ep+1):
        out = bn.forward(X, training=True)
        loss = np.mean((out - true) ** 2)
        
        dout = 2.0 * (out - true) / (B * D)
        dx = bn.backward(dout)
        
        bn.gamma -= lr * bn.dgamma
        bn.beta -= lr * bn.dbeta
        print(f"epoch {ep+1}; loss: {loss:.4f}; gamma: {bn.gamma}; beta: {bn.beta}")
        
    print("\ninference:")
    X_test = np.random.randn(1, D) * 10.0 + 5.0
    out_test = bn.forward(X_test, training=False)
    print("X_test", X_test)
    print("out_test:", out_test)
