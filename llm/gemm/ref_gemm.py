# ref_gemm.py -- PyTorch check of the math the GEMM epilogues implement (GELU-tanh forward and derivative), and of the
# layout conventions (NN / NT / TN) on a small problem emulating bf16 inputs + fp32 accumulation.
# The full-size GEMM verification against cuBLAS and an fp32 kernel reference is in test_gemm.cu.
import math, torch
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = (torch.rand(1 << 20, device=dev) * 16 - 8).bfloat16().float().requires_grad_(True)
y = torch.nn.functional.gelu(x, approximate="tanh"); y.sum().backward()
k0, k1 = math.sqrt(2 / math.pi), 0.044715
t = torch.tanh(k0 * (x + k1 * x ** 3))
g_ours = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k0 * (1 + 3 * k1 * x * x)
y_ours = 0.5 * x * (1 + t)
print("gelu fwd  max abs diff", (y_ours - y).abs().max().item())
print("gelu grad max abs diff", (g_ours - x.grad).abs().max().item())
M, N, K = 256, 128, 64
f = lambda *s: torch.randn(*s, device=dev).bfloat16().float()
X, W, dY = f(M, K), f(K, N), f(M, N)
Y = X @ W                      # NN : A[M,K], B[K,N]
dX = dY @ W.T                  # NT : A=dY[M,N], B=W[K,N] stored [Nout=K, Kred=N]
dW = X.T @ dY                  # TN : A=X stored [K=M, M'=K], B=dY[M,N]
print("shapes", Y.shape, dX.shape, dW.shape)
ok = (y_ours - y).abs().max() < 1e-5 and (g_ours - x.grad).abs().max() < 1e-5
print("PASS" if ok else "FAIL")
