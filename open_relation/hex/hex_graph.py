class Graph:
    def __init__(self, E_h, E_e):
        self.E_h = E_h
        self.E_e = E_e
        self.num_v = E_h.size()[0]
        # Check if graph is consistent
        self.check_consistency()

        # Set up a HEX Graph for forward pass and backward pass
        # Step1: sparsify and densify the graph
        self.sparsify_and_densify()

        # Step2: build junction tree and record
        self.triangularize()
        self.max_span_tree()

    def check_consistency(self):
        # TODO: 检查一致性（dead node）
        raise NotImplementedError

    def sparsify_and_densify(self):
        # TODO: 构建等价稠密图和稀疏图
        # return None
        raise NotImplementedError

    def triangularize(self):
        # TODO: 构建联合图（贝叶斯网）
        raise NotImplementedError

    def max_span_tree(self):
        # TODO: 最大生成树
        raise NotImplementedError

    def list_state_space(self):
        # TODO: 穷举所有合法的多标签分类结果
        # Binary Vector
        raise NotImplementedError

    def record_sumprod(self):
        # TODO:
        raise NotImplementedError