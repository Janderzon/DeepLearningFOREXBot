import tensorflow as tf

def Shallow_LSTM():
    #Define shallow LSTM model.
    shallow_LSTM = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(12, return_sequences=True),
        tf.keras.layers.LSTM(12, return_sequences=False),
        tf.keras.layers.Dense(24, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([24, 1])
    ])

    return shallow_LSTM, "Shallow LSTM"

def Save_Model(model, path):
    tf.keras.models.save_model(
        model,
        path,
        overwrite=False,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )