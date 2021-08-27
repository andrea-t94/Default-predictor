#model structure configuration
model_config = dict(
    model="baseline-default-predictor",
    dense=16,
    dropout=0.2,
    batch_size=128,
    epochs=5,
    learning_rate=1e-3,
    class_weight={0: 0.5, 1: 300} #class weight dictionary, given an unbalacend dataset  88688:1289 = 100:1.43
)

logistic_config = dict(
    solver = 'lbfgs',
    verbose = 1,
    n_jobs = 4,
    class_weight = {0: 0.5, 1: 30}
)
"""
from tensowrflow documentation
weight for class 0 = (1 / neg) * (total / 2.0)
weight for 1 = (1 / pos) * (total / 2.0)
"""