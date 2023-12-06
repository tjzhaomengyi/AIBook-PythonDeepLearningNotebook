# -*- coding: utf-8 -*-
__author__ = 'Mike'
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import keras_tuner as kt

def build_model(hp):
    #从hp对象中对超参数进行采样。采样得到的这些值比如这里的uints变量知识普通的python变量
    units = hp.Int(name="uints", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    #超参数类型可以使不同的类型：Int、Float、Boolean或者Choice
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class SimpleMLP(kt.HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        # 从hp对象中对超参数进行采样。采样得到的这些值比如这里的uints变量知识普通的python变量
        units = hp.Int(name="uints", min_value=16, max_value=64, step=16)
        model = keras.Sequential([
            layers.Dense(unints, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        # 超参数类型可以使不同的类型：Int、Float、Boolean或者Choice
        optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

#我们使用验证集来找到最佳训练轮数
def get_best_epoch(hp):
    model = build_model(hp)
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10)
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch

def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)
    model = build_model(hp)
    model.fit(
        x_train_full, y_train_full,
        batch_size=128, epochs=int(best_epoch * 1.2))
    return model


hypermodel = SimpleMLP(num_classes=10)
#定义一个调节器，其内部可以看做一个for循环，重复以下操作：（1）挑选一组超参数值。（2）使用这些值调用模型构建函数来创建一个模型（3）训练模型并记录模型指标
#build_model指定模型构建函数，或者hypermodel实例。objective:指定调节器要优化的指标，一定要指定校验指标，因为搜索过程的目的是找到能够泛化的模型
# max_trails在结束搜索之前尝试不同模型配置的最大次数。executions_per_trial为了减小指标方差，可以多次训练同意模型并对结果取平均。这个参数是对每种模型配置实验的训练次数
# overwrite开始搜索时是否覆盖目录中的数据，如果修改了模型构建函数，将其设置为True，否则为falose，以便恢复之前启动的使用同意模型构建函数的搜索
tuner = kt.BayesianOptimization(build_model, objective="val_accuracy", max_trials=100, executions_per_trial=2, directory="mnist_kt_test",overwrite=True)
#对于内容指标，比如上述实例中的精度，指标的优化方向（精度应该是最大化，损失是最小化）
# objective = kt.Objective(name="val_accuracy", direction="max")
print(tuner.search_space_summary())
'''
Search space summary
Default search space size: 2
uints (Int)
{'default': None, 'conditions': [], 'min_value': 16, 'max_value': 64, 'step': 16, 'sampling': 'linear'}
optimizer (Choice)
{'default': 'rmsprop', 'conditions': [], 'values': ['rmsprop', 'adam'], 'ordered': False}
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255
x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
x_train_full = x_train[:]
y_train_full = y_train[:]
num_val_samples = 10000
x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]
y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
]
tuner.search(
    x_train, y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2,
)

top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)

best_models = []
for hp in best_hps:
    model = get_best_trained_model(hp)
    model.evaluate(x_test, y_test)
    best_models.append(model)

#如果认为性能略微降低不是大问题，可以选择一条捷径：使用调节器重新加载在超参数搜索过程中保存的具有最佳权重的高性能模型，而无需从头开始重新训练洗的模型
best_models = tuner.get_best_models(top_n)

