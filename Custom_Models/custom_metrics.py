from custom_loss import*

tf.random.set_seed(42)
precision = tf.keras.metrics.Precision()
# print(precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1]))
# precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
# print(precision.result())
# print(precision.variables)

class HuberMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)  
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
tf.random.set_seed(42)
model_metric = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1)
])
model_metric.compile(loss=create_huber(2.0), optimizer="nadam",
                     metrics=[HuberMetric(2.0)])
# model_metric.fit(X_train_scaled, Y_train, epochs=2)

class HuberMetric2(tf.keras.metrics.Mean):
    def __init__(self, threshold=1.0, name = 'HuberMetric2', dtype=None):
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        super(HuberMetric2, self).update_state(metric, sample_weight)
    
    def get_config(self):
        base_config= super().get_config()
        return {**base_config, "threshold": self.threshold}
model_metric2 = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1)
])
model_metric2.compile(loss=create_huber(2.0), optimizer="nadam",
                     metrics=[HuberMetric2(2.0)])
# model_metric2.fit(X_train_scaled, Y_train, epochs=2)