import tensorflow as tf
from tensorflow.keras import backend as K

def compute_grad_cam(model, input_volume, class_index, layer_name):
    """
    Returns CAM of shape [H, W, D] in [0,1] for the given class_index.
    Works for per-voxel softmax outputs (segmentation).
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(input_volume, training=False)   # conv_out: [B,h',w',d',C]
        # For segmentation heads: preds is [B,H,W,D,n_classes]
        # Use spatial mean for the class map as the scalar loss:
        loss = tf.reduce_mean(preds[..., class_index])

    grads = tape.gradient(loss, conv_out)                            # [B,h',w',d',C]
    weights = tf.reduce_mean(grads, axis=(1, 2, 3, 4), keepdims=True)# [B,1,1,1,1,C]
    cam = tf.reduce_sum(weights * conv_out, axis=-1)                 # [B,h',w',d']
    cam = tf.nn.relu(cam)[0].numpy()                                 # [h',w',d']

    # Trilinear upsample to input size
    H, W, D = input_volume.shape[1:4]
    zf = (H / cam.shape[0], W / cam.shape[1], D / cam.shape[2])      # scale (y, x, z)
    cam = zoom(cam, zf, order=1)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam  # [H,W,D]

def compute_grad_cam_iou(cam_mask, ground_truth, threshold=0.5):
    cam_bin = (cam_mask > threshold).astype(np.uint8)
    gt_bin = (ground_truth > 0.5).astype(np.uint8)
    intersection = np.sum(cam_bin * gt_bin)
    union = np.sum(cam_bin) + np.sum(gt_bin) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_average_grad_cam_iou(model, val_X, val_y, layer_name='conv_last'):
    batch_size = val_X.shape[0]
    n_classes = val_y.shape[-1]
    iou_scores = []

    for i in range(batch_size):
        input_vol = val_X[i:i+1]
        gt_mask = val_y[i]
        for cls in range(n_classes):
            gt_cls_mask = gt_mask[..., cls]
            try:
                cam = compute_grad_cam(model, input_vol, cls, layer_name=layer_name)  # [H,W,D]
            except Exception:
                # layer missing or shape mismatch → penalize gracefully
                iou_scores.append(0.0)
                continue
            iou = compute_grad_cam_iou(cam, gt_cls_mask)
            iou_scores.append(iou)

    return float(np.mean(iou_scores)) if iou_scores else 0.0

# --------------- Binary metrics & wrappers ---------------
def dice_binary(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)





def class_metric(metric_fn, class_index, name):
    def metric(y_true, y_pred):
        y_true_c = y_true[..., class_index]
        y_pred_c = y_pred[..., class_index]
        return metric_fn(y_true_c, y_pred_c)
    metric.__name__ = name
    return metric

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    TP = tf.reduce_sum(y_true * y_pred, axis=[1,2,3,4])
    FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2,3,4])
    FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2,3,4])
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return tf.reduce_mean(1.0 - tversky)

# metrics list + monitored metric
metrics = []
for c, name in enumerate(['wt', 'tc', 'et']):
    metrics += [
        class_metric(dice_binary, c, f'dice_{name}'),
        class_metric(iou_binary, c, f'iou_{name}'),
        class_metric(precision_binary, c, f'precision_{name}'),
        class_metric(recall_binary, c, f'recall_{name}'),
        class_metric(f1_binary, c, f'f1_{name}'),
        class_metric(accuracy_binary, c, f'accuracy_{name}')
    ]
# add dice_wt explicitly (monitor key must exist)
metrics.append(class_metric(dice_binary, 0, 'dice_wt'))