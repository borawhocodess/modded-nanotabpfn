class Prior:
    activations = (lambda z: z, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)

    def __init__(self, config, device):
        self.config = config
        self.device = device
        assert self.config.max_num_test_rows < self.config.min_num_rows

    def hyperparameters(self):
        c = self.config
        self.num_cols = int(np.random.randint(c.min_num_cols, c.max_num_cols + 1))
        self.num_rows = int(np.random.randint(c.min_num_rows, c.max_num_rows + 1))
        self.nodes = self.num_cols + 1
        self.num_test_rows = int(np.random.randint(c.min_num_test_rows, c.max_num_test_rows + 1))
        self.sep = self.num_rows - self.num_test_rows
        self.redirection = np.random.uniform(c.min_redirection, c.max_redirection)
        self.num_classes = int(np.random.randint(c.min_num_classes, c.max_num_classes + 1))
        self.num_parent_attempts = int(np.random.randint(c.min_num_parent_attempts, c.max_num_parent_attempts + 1))

    def gnr(self):
        parents = [[] for _ in range(self.nodes)]
        for child in range(1, self.nodes):
            chosen = set()
            for _ in range(self.num_parent_attempts):
                candidate = int(np.random.randint(child))
                if np.random.rand() < self.redirection and parents[candidate]:
                    candidate = int(np.random.choice(parents[candidate]))
                chosen.add(candidate)
            parents[child] = sorted(chosen)
        return parents

    def propagate(self):
        parents = self.gnr()
        w = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for i in range(1, self.nodes):
            w[i, parents[i]] = np.random.randn(len(parents[i]))
        w = torch.from_numpy(w).to(self.device)
        acts = np.random.randint(len(self.activations), size=self.nodes)
        z = torch.randn(self.num_rows, self.nodes, device=self.device)
        for i in range(1, self.nodes):
            zi = self.activations[acts[i]](z @ w[i]) + 0.1 * z[:, i]
            std, mean = torch.std_mean(zi)
            z[:, i] = (zi - mean) / (std + 1e-6)
        return z

    def target(self, z):
        target = int(np.random.randint(1, self.nodes))
        zt = z[:, target].contiguous()
        cuts = torch.linspace(0, 1, self.num_classes + 1, device=self.device)[1:-1]
        y = torch.bucketize(zt, zt.quantile(cuts))
        x = torch.cat([z[:, :target], z[:, target + 1 :]], dim=1)
        return x, y.float()

    def postprocess(self, x):
        return x

    def dataset(self):
        z = self.propagate()
        x, y = self.target(z)
        x = self.postprocess(x)
        return x, y

    def batch(self, batch_size):
        self.hyperparameters()
        datasets = [self.dataset() for _ in range(batch_size)]
        x = torch.stack([d[0] for d in datasets])
        y = torch.stack([d[1] for d in datasets])
        sep = self.sep
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]
