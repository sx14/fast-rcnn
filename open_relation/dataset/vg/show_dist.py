import os
import matplotlib.pyplot as plt
import numpy as np
from open_relation.dataset.dataset_config import DatasetConfig


dataset_config = DatasetConfig('vg')
target = 'predicate'   # TODO modify


# counter
label_counter = dict()

# counting








plt.plot(ranked_labels, ranked_counts)
plt.title('distribution')
plt.xlabel('label')
plt.ylabel('count')
plt.show()
