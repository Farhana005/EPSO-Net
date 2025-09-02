import os
import json
import random
import numpy as np
import tensorflow as tf
from train import train
from model import best_model, DepthwiseSeparableConv3D, GroupNormalization, get_flops
from tensorflow.keras.layers import Conv3D, BatchNormalization
from metrics import tversky_loss, metrics
from medpy.metric.binary import hd95

search_space = {    
    "num_layers": range(1, 4),  # [1, 2, 3]
    "kernel_size": [3, 5, 7],   
    "dilation_rate": [1, 2, 3, 4, 5], 
    "filters": range(16, 128),  # [16, ..., 128]
    "activation": [relu, tanh, sigmoid, LeakyReLU],
    "dropout_rate": lambda: round(random.uniform(0.2, 0.5), 2),  # e.g., 0.21, 0.35
    "select_conv": [DepthwiseSeparableConv3D, Conv3D],  # Conv layer choices
    "select_norm": [LayerNormalization, GroupNormalization, BatchNormalization],  # Normalization types
    "pooling_type": [MaxPooling3D, AveragePooling3D],
    "aggregation_method": [Add, Concatenate, Multiply],
}

def random_architecture(search_space):
    return {
        "num_layers": random.choice(search_space["num_layers"]),
        "kernel_size": random.choice(search_space["kernel_size"]),
        "dilation_rate": random.choice(search_space["dilation_rate"]),
        "filters": random.choice(search_space["filters"]),
        "activation": random.choice(search_space["activation"]),
        "dropout_rate": search_space["dropout_rate"](),  # Call lambda
        "select_conv": random.choice(search_space["select_conv"]),
        "select_norm": random.choice(search_space["select_norm"]),
        "pooling_type": random.choice(search_space["pooling_type"]),
        "aggregation_method": random.choice(search_space["aggregation_method"]),
    }

class PSOController:
    def __init__(self, w=0.7, c1=1.6, c2=1.6, vmax=0.15, bounds=(0.02, 0.40), constriction=True):
        self.w, self.c1, self.c2 = w, c1, c2
        self.vmax = float(vmax)
        self.bounds = bounds
        self.constriction = constriction
        self.state = {}  # lineage_id -> {'x','v','pbest','pbest_fit'}

    def _chi(self):
        phi = self.c1 + self.c2
        if phi <= 4:  # χ only defined for phi>4
            return 1.0
        return 2.0 / (abs(2 - phi - math.sqrt(phi**2 - 4*phi)))

    def ensure(self, lid, init_x):
        if lid not in self.state:
            self.state[lid] = {"x": float(init_x), "v": 0.0,
                               "pbest": float(init_x), "pbest_fit": -np.inf}
        return self.state[lid]

    def suggest(self, lid, parent_mu1, parent_mu2, gbest_rate):
        x_i = 0.5 * (parent_mu1 + parent_mu2)  # seed near parents
        s = self.ensure(lid, x_i)
        r1, r2 = random.random(), random.random()
        v_new = ( self.w * s["v"]
                  + self.c1 * r1 * (s["pbest"] - s["x"])
                  + self.c2 * r2 * (float(gbest_rate) - s["x"]) )
        if self.constriction:
            v_new *= self._chi()
        v_new = float(np.clip(v_new, -self.vmax, self.vmax))
        x_new = float(np.clip(s["x"] + v_new, self.bounds[0], self.bounds[1]))
        s["_next_x"], s["_next_v"] = x_new, v_new
        return x_new

    def update(self, lid, achieved_fitness):
        s = self.state[lid]
        s["x"], s["v"] = s.pop("_next_x"), s.pop("_next_v")
        if achieved_fitness > s["pbest_fit"]:
            s["pbest_fit"] = float(achieved_fitness)
            s["pbest"] = float(s["x"])


# ---------- crossover uses PSO to set child's rate ----------
def crossover_and_mutate(parent1, parent2,
                         global_best_mutation_rate,
                         search_space,
                         pso: PSOController,
                         child_lineage_id: str,
                         default_mutation_rate=0.1):
    """Create child by uniform crossover, then let PSO 'suggest' its mutation_rate.
       NOTE: Call pso.update(child_lineage_id, fitness) AFTER evaluation.
    """
    child = {}

    # 1) Uniform crossover of all genes (child gets its own lineage id)
    for key in parent1:
        # we'll set mutation_rate separately via PSO; skip if not in both parents
        child[key] = random.choice([parent1[key], parent2[key]])
    child["id"] = child_lineage_id

    # 2) Parent rates (fallback to default)
    mu1 = parent1.get("mutation_rate", default_mutation_rate)
    mu2 = parent2.get("mutation_rate", default_mutation_rate)

    # 3) TRUE PSO: suggest the child's next rate from the particle dynamics
    child_rate = pso.suggest(child_lineage_id, mu1, mu2, gbest_rate=global_best_mutation_rate)
    child["mutation_rate"] = float(child_rate)

    # 4) Mutate other genes using the PSO-driven rate
    for key in list(child.keys()):
        if key in ("mutation_rate", "id"):
            continue
        if key in search_space and random.random() < child["mutation_rate"]:
            pool = search_space[key]
            child[key] = pool() if callable(pool) else random.choice(list(pool))

    return child


class PSOController:
    def __init__(self, w=0.7, c1=1.6, c2=1.6, vmax=0.15, bounds=(0.02, 0.40), constriction=True):
        self.w, self.c1, self.c2 = w, c1, c2
        self.vmax = float(vmax)
        self.bounds = bounds
        self.constriction = constriction
        self.state = {}  # lineage_id -> {'x','v','pbest','pbest_fit'}

    def _chi(self):
        phi = self.c1 + self.c2
        if phi <= 4:  # χ only defined for phi>4
            return 1.0
        return 2.0 / (abs(2 - phi - math.sqrt(phi**2 - 4*phi)))

    def ensure(self, lid, init_x):
        if lid not in self.state:
            self.state[lid] = {"x": float(init_x), "v": 0.0,
                               "pbest": float(init_x), "pbest_fit": -np.inf}
        return self.state[lid]

    def suggest(self, lid, parent_mu1, parent_mu2, gbest_rate):
        x_i = 0.5 * (parent_mu1 + parent_mu2)  # seed near parents
        s = self.ensure(lid, x_i)
        r1, r2 = random.random(), random.random()
        v_new = ( self.w * s["v"]
                  + self.c1 * r1 * (s["pbest"] - s["x"])
                  + self.c2 * r2 * (float(gbest_rate) - s["x"]) )
        if self.constriction:
            v_new *= self._chi()
        v_new = float(np.clip(v_new, -self.vmax, self.vmax))
        x_new = float(np.clip(s["x"] + v_new, self.bounds[0], self.bounds[1]))
        s["_next_x"], s["_next_v"] = x_new, v_new
        return x_new

    def update(self, lid, achieved_fitness):
        s = self.state[lid]
        s["x"], s["v"] = s.pop("_next_x"), s.pop("_next_v")
        if achieved_fitness > s["pbest_fit"]:
            s["pbest_fit"] = float(achieved_fitness)
            s["pbest"] = float(s["x"])


# ---------- crossover uses TRUE PSO to set child's rate ----------
def crossover_and_mutate(parent1, parent2,
                         global_best_mutation_rate,
                         search_space,
                         pso: PSOController,
                         child_lineage_id: str,
                         default_mutation_rate=0.1):
    """Create child by uniform crossover, then let PSO 'suggest' its mutation_rate.
       NOTE: Call pso.update(child_lineage_id, fitness) AFTER evaluation.
    """
    child = {}

    # 1) Uniform crossover of all genes (child gets its own lineage id)
    for key in parent1:
        # we'll set mutation_rate separately via PSO; skip if not in both parents
        child[key] = random.choice([parent1[key], parent2[key]])
    child["id"] = child_lineage_id

    # 2) Parent rates (fallback to default)
    mu1 = parent1.get("mutation_rate", default_mutation_rate)
    mu2 = parent2.get("mutation_rate", default_mutation_rate)

    # 3) TRUE PSO: suggest the child's next rate from the particle dynamics
    child_rate = pso.suggest(child_lineage_id, mu1, mu2, gbest_rate=global_best_mutation_rate)
    child["mutation_rate"] = float(child_rate)

    # 4) Mutate other genes using the PSO-driven rate
    for key in list(child.keys()):
        if key in ("mutation_rate", "id"):
            continue
        if key in search_space and random.random() < child["mutation_rate"]:
            pool = search_space[key]
            child[key] = pool() if callable(pool) else random.choice(list(pool))

    return child

def evaluate_fitness(individual, training_generator, valid_generator, num_epochs, n_channels,
                     generation=None, individual_index=None):
    model = best_model(
        input_shape=(128, 128, 128, n_channels),
        num_layers=individual['num_layers'],
        dilation_rate=individual['dilation_rate'],
        filters=individual['filters'],
        kernel_size=individual['kernel_size'],
        activation=individual['activation'],
        dropout_rate=individual['dropout_rate'],
        select_conv=individual['select_conv'],
        select_norm=individual['select_norm'],
        pooling_type=individual['pooling_type'],
        aggregation_method=individual['aggregation_method'],
        n_classes=3
    )

    model.compile(optimizer=Adam(1e-4), loss=tversky_loss, metrics=metrics)

    arch_id = f"gen{generation}_ind{individual_index}"
    ckpt_dir = os.path.join("checkpoints", arch_id); os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, "ckpt.weights.h5")

    early_stop = EarlyStopping(monitor='val_dice_wt', patience=5, mode='max', restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_dice_wt', factor=0.5, patience=3, mode='max')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_dice_wt',
                                 save_best_only=True, mode='max', save_weights_only=True)

    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            print(f"[INFO] Resumed weights from {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}")

    history = model.fit(
        training_generator,
        validation_data=valid_generator,
        epochs=num_epochs,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=0
    )

    # pull a validation batch robustly
    try:
        val_X, val_y_true = valid_generator[0]
    except Exception:
        val_X, val_y_true = next(iter(valid_generator))

    val_y_pred = model.predict(val_X, verbose=0)
    y_true_class = tf.argmax(val_y_true, axis=-1).numpy()
    y_pred_class = tf.argmax(val_y_pred, axis=-1).numpy()

    # avg HD95 (classes 1..3)
    def compute_hd95_average(y_pred, y_true):
        from medpy.metric.binary import hd95
        hd95s = []
        for cls in range(1, 4):
            pred_bin = (y_pred == cls).astype(np.uint8)
            true_bin = (y_true == cls).astype(np.uint8)
            try:
                hd = hd95(pred_bin, true_bin)
            except Exception:
                hd = 10.0
            hd95s.append(hd)
        return float(np.mean(hd95s)) if hd95s else 10.0

    avg_hd95 = compute_hd95_average(y_pred_class, y_true_class)
    params = model.count_params() / 1e6
    flops = get_flops(model) / 1e9

    # Grad-CAM IoU (guard if layer missing)
    try:
        gradcam_iou = compute_average_grad_cam_iou(model, val_X, val_y_true, layer_name='conv_last')
    except Exception:
        gradcam_iou = 0.0

    val_dice = history.history.get("val_dice_wt", [0.0])[-1]
    fitness_score = custom_score(val_dice, avg_hd95, params, flops, gradcam_iou)

    # save per-epoch logs
    os.makedirs("metrics_logs", exist_ok=True)
    training_log = []
    for epoch in range(len(history.history['loss'])):
        row = {'epoch': epoch + 1}
        for key, arr in history.history.items():
            row[key] = float(arr[epoch])
        training_log.append(row)
    with open(f"metrics_logs/gen{generation}_ind{individual_index}.json", "w") as f:
        json.dump(training_log, f, indent=4)

    return fitness_score, {
        "dice": float(val_dice),
        "hd95": float(avg_hd95),
        "params": float(params),
        "flops": float(flops),
        "gradcam_iou": float(gradcam_iou),
        "score": float(fitness_score),
        "mutation_rate": float(individual.get("mutation_rate", 0.1))
    }
