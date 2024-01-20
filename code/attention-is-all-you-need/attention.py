import numpy as np
import math


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


class ScaledDotProductAttention:
    """
    This is a simple class which demonstrates how we can implement Scaled Dot-Product Attention using numpy. It's not optimised at all.
    """

    def __init__(self, d_k: int, d_v: int, d_input: int) -> None:
        """
        We initialise our Q,K and V matrices

        Note here that

        d_k : Dimensionality of K and Q weight matrix
        d_v : Dimensionality of V weight matrix
        d_input : Dimensionality of embedding layer
        """
        self.w_q = np.random.randn(d_input, d_k)
        self.w_k = np.random.randn(d_input, d_k)
        self.w_v = np.random.randn(d_input, d_v)
        self.d_k = d_k
        self.d_input = d_input

    def __call__(self, x: np.array, mask=False) -> np.array:
        """
        This is how we can call our forward pass function
        """
        assert (
            x.shape[1] == self.d_input
        ), f"Passed in an array of size {len(x)}. Maximum context size is {self.d_input}"
        q_prime = x @ self.w_q.T  # x : (d_input , d_k ) , w_q.T : (d_k , d_input )
        k_prime = x @ self.w_k.T  # x : (d_input , d_k ) , w_k.T : (d_k , d_input )
        v_prime = x @ self.w_v.T

        qk = (q_prime @ k_prime.T) / math.sqrt(self.d_k)  # d_input x  d_input

        if mask:
            mask = np.tril(np.ones_like(qk))
            mask[mask == 0] = -np.infty
            mask[mask == 1] = 0
            print(mask)
            qk += mask

        qk_normalised = softmax(qk)
        return qk_normalised @ v_prime


if __name__ == "__main__":
    d_model = 4
    d_q = 4
    d_v = 4
    AttentionOperation = ScaledDotProductAttention(
        d_q, d_v, d_model
    )  # We perform a simple
    x = np.random.randn(3, d_model)
    res = AttentionOperation(x)
    assert res.shape == (3, 4)
