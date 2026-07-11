import torch
import pytest
from harvester.v25.solver import BatchedSylvesterSolver, make_laplacian_1d

def test_laplacian_construction():
    """Verify that the 1D tridiagonal Laplacian is constructed with Dirichlet boundaries."""
    L = make_laplacian_1d(5)
    assert L.shape == (5, 5)
    # Check tridiagonal values
    assert L[0, 0] == 2.0
    assert L[0, 1] == -1.0
    assert L[1, 0] == -1.0
    assert L[2, 2] == 2.0

def test_sylvester_mathematical_correctness():
    """Verify that (I + gamma_c L_c) X + X (gamma_r L_r) = Y is satisfied by the solver output."""
    B, H, W = 2, 8, 8
    solver = BatchedSylvesterSolver(H, W)
    
    Y = torch.randn(B, H, W)
    gamma_c = torch.tensor([0.5, 1.5]).view(B, 1, 1)
    gamma_r = torch.tensor([0.2, 0.8]).view(B, 1, 1)
    
    X = solver.solve(Y, gamma_c, gamma_r)
    assert X.shape == (B, H, W)
    
    # Verify the equation holds for each batch item
    L_c = make_laplacian_1d(H)
    L_r = make_laplacian_1d(W)
    
    for b in range(B):
        X_b = X[b]
        Y_b = Y[b]
        g_c = gamma_c[b, 0, 0]
        g_r = gamma_r[b, 0, 0]
        
        # LHS: (I + g_c L_c) X + X (g_r L_r)
        term1 = torch.matmul(torch.eye(H) + g_c * L_c, X_b)
        term2 = torch.matmul(X_b, g_r * L_r)
        LHS = term1 + term2
        
        # Assert that LHS is close to Y_b
        torch.testing.assert_close(LHS, Y_b, rtol=1e-4, atol=1e-4)

def test_solver_gradients():
    """Verify that gradients flow correctly through the solver to gamma inputs."""
    H, W = 8, 8
    solver = BatchedSylvesterSolver(H, W)
    
    Y = torch.randn(1, H, W)
    gamma_c_leaf = torch.tensor([1.0], requires_grad=True)
    gamma_r_leaf = torch.tensor([1.0], requires_grad=True)
    gamma_c = gamma_c_leaf.view(1, 1, 1)
    gamma_r = gamma_r_leaf.view(1, 1, 1)
    
    X = solver.solve(Y, gamma_c, gamma_r)
    loss = X.sum()
    loss.backward()
    
    assert gamma_c_leaf.grad is not None
    assert gamma_r_leaf.grad is not None
