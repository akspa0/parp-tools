import torch
import torch.nn as nn

def make_laplacian_1d(n, device="cpu"):
    """Construct a standard 1D Laplacian matrix with Dirichlet boundary conditions."""
    diag = 2.0 * torch.ones(n, device=device)
    offdiag = -1.0 * torch.ones(n - 1, device=device)
    L = torch.diag(diag) + torch.diag(offdiag, 1) + torch.diag(offdiag, -1)
    return L

class BatchedSylvesterSolver:
    """Solves the Sylvester equation (I + gamma_c L_c) X + X (gamma_r L_r) = Y on PyTorch GPU."""
    def __init__(self, h, w, device="cpu"):
        self.h = h
        self.w = w
        self.device = device
        
        # Precompute Laplacians and perform eigendecomposition
        L_c = make_laplacian_1d(h, device=device)
        L_r = make_laplacian_1d(w, device=device)
        
        # symmetric matrices yield real eigenvalues and orthogonal eigenvectors
        self.D_c, self.U_c = torch.linalg.eigh(L_c)
        self.D_r, self.U_r = torch.linalg.eigh(L_r)
        
    def to(self, device):
        """Move the solver parameters to the specified device."""
        self.device = device
        self.D_c = self.D_c.to(device)
        self.U_c = self.U_c.to(device)
        self.D_r = self.D_r.to(device)
        self.U_r = self.U_r.to(device)
        return self
        
    def solve(self, Y, gamma_c, gamma_r):
        """Solve (I + gamma_c L_c) X + X (gamma_r L_r) = Y for X.
        
        Args:
            Y: Tensor of shape (B, H, W)
            gamma_c: Tensor of shape (B, 1, 1) or scalar parameter
            gamma_r: Tensor of shape (B, 1, 1) or scalar parameter
            
        Returns:
            X: Solved Tensor of shape (B, H, W)
        """
        device = Y.device
        U_c = self.U_c.to(device)
        U_r = self.U_r.to(device)
        D_c = self.D_c.to(device)
        D_r = self.D_r.to(device)
        
        # Diagonalize Y: Y_tilde = U_c^T @ Y @ U_r
        Y_tilde = torch.matmul(U_c.t(), torch.matmul(Y, U_r))
        
        # Reshape D_c and D_r for broadcasting
        # D_c_exp: (1, H, 1)
        # D_r_exp: (1, 1, W)
        D_c_exp = D_c.view(1, -1, 1)
        D_r_exp = D_r.view(1, 1, -1)
        
        # Ensure gamma tensors are reshaped for broadcasting
        if isinstance(gamma_c, torch.Tensor) and gamma_c.dim() < 3:
            gamma_c = gamma_c.view(-1, 1, 1)
        if isinstance(gamma_r, torch.Tensor) and gamma_r.dim() < 3:
            gamma_r = gamma_r.view(-1, 1, 1)
            
        # Denominator: 1 + gamma_c * D_c + gamma_r * D_r
        denom = 1.0 + gamma_c * D_c_exp + gamma_r * D_r_exp
        
        X_tilde = Y_tilde / denom
        
        # Reconstruct X: X = U_c @ X_tilde @ U_r^T
        X = torch.matmul(U_c, torch.matmul(X_tilde, U_r.t()))
        return X
