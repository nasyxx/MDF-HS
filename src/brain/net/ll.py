class ATT(hk.Module):
    """Attention module."""

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        """Initialize the module."""
        super().__init__(name=name)
        self.l = hk.Linear(1)

    def __call__(self, adj: Array, x: Array) -> Array:
        """Forward."""
        n = x.shape[0]
        x = x.reshape(n, 1, -1)
        x_left = jnp.tile(x[:, None], (1, n, 1))
        x_right = jnp.tile(x[None], (n, 1, 1))
        x_cat = jnp.concatenate([x_left, x_right], axis=-1)
        att_adj = self.l(x_cat).squeeze()
        att_adj = jax.nn.sigmoid(att_adj)
        return adj @ att_adj
