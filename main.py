import tensorflow as tf
import gc
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from functools import reduce
from operator import mul

# To decay learning rate for smooth convergence
class ExponentialDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay_factor=0.95):
        super(ExponentialDecayCallback, self).__init__()
        self.decay_factor = decay_factor

    def on_epoch_end(self, epoch, logs=None):
        # Update learning rate
        lr = self.model.optimizer.learning_rate.numpy()
        lr *= self.decay_factor
        self.model.optimizer.learning_rate.assign(lr)



def sort_by_highest_product(list_of_lists):
    products = [(lst, reduce(mul, lst, 1)) for lst in list_of_lists]
    sorted_lists = [lst for lst, _ in sorted(products, key=lambda x: x[1], reverse=True)]
    return sorted_lists

def hyperparam_tuner(X_train, y_train, X_val, y_val, total_outputs, neuron_combinations_array, loss_func, task, learning_rates=[1., 0.1, 0.01, 0.001, 0.0001], lr_decay=1, reg_lambdas=[1., 0.1, 0.01, 0.001, 0.0001], epochs=50, batch_size=128, batch_norm=True, smart_skipping=True, verbose=0):
    '''
    neuron_combination_array: for manually entering the layers and neurons in each hidden layer. Enclose the neurons of different layers in separate lists, enclosed in an outer list.
                              Example: A combination of 1 hidden layer neurons: 2,4,8 and 2 hidden layer neurons [16, 8], [8, 8], [8, 4] will be represented as:
                              [ [ [2], [4], [8] ], [ [16, 8], [8, 8], [8, 4] ] ]
    task: 'classification' or 'regression'
    lr_decay: decays learning rate after every epoch. If lr_decay=0.95 then after every epoch lr = 0.95 * lr 
    '''
    if task != 'classification' and task != 'regression':
      raise Exception("task should be either 'classification' or 'regression'")
    if lr_decay > 1:
       raise Exception("lr_decay must be less than or equal to 1")
    trial_count = 0
    learning_rates = sorted(learning_rates, reverse=True) #Ensuring the learning rates in descending order
    models = []
    neuron_combinations_sum = 0
    for layer in neuron_combinations_array:
        neuron_combinations_sum += len(layer)
    print(f'Total Trials: {neuron_combinations_sum * len(learning_rates) * len(reg_lambdas) * (2 if batch_norm else 1)}\n')
    for layer in range(1, len(neuron_combinations_array)+1):
        neuron_combinations = neuron_combinations_array[layer-1]
        neuron_combinations = sort_by_highest_product(neuron_combinations)  # Sorting to ensure that complex model is built first
        neuron_combination_idx = 0
        skip_batch_norm = False
        skip_learning_rates = [False for i in range(len(learning_rates))]
        skip_reg_lambdas = [False for i in range(len(reg_lambdas))]
        while neuron_combination_idx < len(neuron_combinations):
            models_dev = []
            for batch_norm in range(2 if batch_norm else 1):
              if batch_norm == 1 and skip_batch_norm:
                for i in range(len(learning_rates) * len(reg_lambdas)):
                  trial_count += 1
                  print(f'Trial #{trial_count}\n')
                  print('Skipped by Smart Skipping! (Batch Normalization Not Required)\n')
                break
              for lambda_idx, reg_lambda in enumerate(reg_lambdas):
                if skip_reg_lambdas[lambda_idx]:
                  for i in range(len(learning_rates)):
                    trial_count += 1
                    print(f'Trial #{trial_count}\n')
                    print('Skipped by Smart Skipping! (Too High Regularization Lambda)\n')
                  continue
                skip_trial = False
                last_val_loss = float('inf')
                for lr_idx, lr in enumerate(learning_rates):
                  trial_count +=1
                  print(f'Trial #{trial_count}\n')
                  if skip_learning_rates[lr_idx]:
                    print('Skipped by Smart Skipping! (Too Slow Learning Rate)\n')
                    continue
                  if not skip_trial:
                    model = tf.keras.Sequential()
                    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
                    for neuron in neuron_combinations[neuron_combination_idx]:
                        if reg_lambda > 0:
                          model.add(tf.keras.layers.Dense(units=neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_lambda)))
                        else:
                          model.add(tf.keras.layers.Dense(units=neuron, activation='relu'))
                        if batch_norm == 1:
                            model.add(tf.keras.layers.BatchNormalization())
                    print(f'Total hidden layers: {len(neuron_combinations[neuron_combination_idx])}')
                    print(f'Neurons combination: {neuron_combinations[neuron_combination_idx]}')
                    print(f'Learning Rate: {lr}')
                    print(f'Regularization Lambda: {reg_lambda}')
                    print(f"Batch Normalization: {'Yes' if batch_norm==1 else 'No'}\n")
                    if reg_lambda > 0:
                      model.add(tf.keras.layers.Dense(total_outputs, kernel_regularizer=tf.keras.regularizers.L2(reg_lambda)))
                    else:
                      model.add(tf.keras.layers.Dense(total_outputs))
                    monitor = ExponentialDecayCallback(decay_factor=lr_decay) if lr_decay < 1 else []
                    model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
                    model.fit(X_train, y_train, callbacks=[monitor], verbose=verbose, epochs=epochs, batch_size=batch_size)
                    val_loss = model.evaluate(X_val, y_val, verbose=0)
                    training_loss = model.evaluate(X_train, y_train, verbose=0)
                    y_pred = model.predict(X_val, verbose=0)
                    models_dev.append({
                        'model': model,
                        'hidden_layers': len(neuron_combinations[neuron_combination_idx]),
                        'hidden_layer_neurons': neuron_combinations[neuron_combination_idx],
                        'learning_rate': lr,
                        'regularization_lambda': reg_lambda,
                        'batch_norm': 'yes' if batch_norm == 1 else 'no',
                        'training_loss': training_loss,
                        'val_loss': val_loss,
                        'lr_decay': lr_decay,
                        'epochs': epochs
                    })
                    if task == "regression":
                      val_mape = mean_absolute_percentage_error(y_val, y_pred)
                      val_mape = round(val_mape*100,2)
                      models_dev[-1]['val_mape'] = val_mape
                    print(f"\nTraining Loss: {training_loss}")
                    print(f"Validation Loss: {val_loss}\n\n")
                    # Clear the session to free memory
                    tf.keras.backend.clear_session(free_memory=True)
                    del model
                    gc.collect()
                    if last_val_loss < val_loss:
                      skip_trial = True
                    last_val_loss = val_loss
                  else:
                    print('Trial skipped because of too slow learning rate! \n\n')
            models_dev = sorted(models_dev, key=lambda x: x['val_loss'])
            models.append(models_dev[0])
            if smart_skipping:
              for i in range(len(learning_rates)):
                if learning_rates[i] < models_dev[0]['learning_rate']:
                  skip_learning_rates[i] = True

              for i in range(len(reg_lambdas)):
                if reg_lambdas[i] < models_dev[0]['regularization_lambda']:
                  skip_reg_lambdas[i] = True

              if models_dev[0]['batch_norm'] == 'no':
                skip_batch_norm = True 
            neuron_combination_idx += 1
    models = sorted(models, key=lambda x: x['val_loss'])
    return models