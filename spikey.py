# PROIECTUL INIȚIAL: https://github.com/frhtas/AI-Dino

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras_spiking
from tensorflow.python.keras.models import Sequential

tf.random.set_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = tf.keras.datasets.fashion_mnist.load_data()

    # normalize images so values are between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    num_classes = len(class_names)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.axis("off")
        plt.title(class_names[train_labels[i]])

    #train(model, train_images, test_images)


    def train(input_model, train_x, test_x):
        input_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        print(train_labels.shape)

        input_model.fit(train_x, train_labels, epochs=10)

        _, test_acc = input_model.evaluate(test_x, test_labels, verbose=2)

        print("\nTest accuracy:", test_acc)

    # repeat the images for n_steps
    n_steps = 3

    print(train_images.shape)
    train_sequences = np.tile(train_images[:, None], (1, n_steps, 1, 1))
    print(train_images[:, None].shape)

    test_sequences = np.tile(test_images[:, None], (1, n_steps, 1, 1))

    train_sequences = tf.expand_dims(train_sequences, axis=-1)



    #print('train shape', train_sequences.shape)
    #print('test shape', test_sequences.shape)

    # THE GOOD ONE
    # spiking_model = tf.keras.Sequential(
    #     [
    #         # add temporal dimension to the input shape; we can set it to None,
    #         # to allow the model to flexibly run for different lengths of time
    #         tf.keras.layers.Reshape((-1, 28 * 28), input_shape=(None, 28, 28)),
    #         # we can use Keras' TimeDistributed wrapper to allow the Dense layer
    #         # to operate on temporal data
    #         tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128)),
    #         # replace the "relu" activation in the non-spiking model with a
    #         # spiking equivalent
    #         keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #         # use average pooling layer to average spiking output over time
    #         tf.keras.layers.GlobalAveragePooling1D(),
    #         tf.keras.layers.Dense(10),
    #     ]
    # )

    model = Sequential()

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 1))))
    model.add(keras_spiking.SpikingActivation("relu", spiking_aware_training=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, kernel_size=(3, 3))))
    model.add(keras_spiking.SpikingActivation("relu", spiking_aware_training=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=(3, 3))))
    model.add(keras_spiking.SpikingActivation("relu", spiking_aware_training=False))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128)))
    model.add(keras_spiking.SpikingActivation("relu", spiking_aware_training=False))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(10))

    spiking_model = model
    # GOOD ENOUGH
    # mzg = tf.keras.Sequential(
    #         [
    #             tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
    #                    input_shape=(28, 28, 1)),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.Conv2D(64, kernel_size=(3, 3)),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #             tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.Conv2D(128, kernel_size=(3, 3)),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #             tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same'),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.Conv2D(256, kernel_size=(3, 3)),
    #             keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #             tf.keras.layers.Flatten(),
    #             tf.keras.layers.Dense(1024, activation='relu'),
    #             tf.keras.layers.Dense(512, activation='relu'),
    #         ]
    #     )
    #
    # spiking_model = tf.keras.Sequential([
    #     tf.keras.layers.TimeDistributed(mzg),
    #
    #     # replace the "relu" activation in the non-spiking model with a
    #     # spiking equivalent
    #     keras_spiking.SpikingActivation("relu", spiking_aware_training=False),
    #     # use average pooling layer to average spiking output over time
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(10),
    # ])


    # train the model, identically to the non-spiking version,
    # except using the time sequences as inputs
    train(spiking_model, train_sequences, test_sequences)


    def check_output(seq_model, modify_dt=None):
        """
        This code is only used for plotting purposes, and isn't necessary to
        understand the rest of this example; feel free to skip it
        if you just want to see the results.
        """

        # rebuild the model with the functional API, so that we can
        # access the output of intermediate layers
        inp = x = tf.keras.Input(batch_shape=seq_model.layers[0].input_shape)

        has_global_average_pooling = False
        for layer in seq_model.layers:
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
                # remove the pooling so that we can see the model's
                # output over time
                has_global_average_pooling = True
                continue

            if isinstance(layer, (keras_spiking.SpikingActivation, keras_spiking.Lowpass)):
                cfg = layer.get_config()
                # update dt, if specified
                if modify_dt is not None:
                    cfg["dt"] = modify_dt
                # always return the full time series so we can visualize it
                cfg["return_sequences"] = True

                layer = type(layer)(**cfg)

            if isinstance(layer, keras_spiking.SpikingActivation):
                # save this layer so we can access it later
                spike_layer = layer

            x = layer(x)

        func_model = tf.keras.Model(inp, [x, spike_layer.output])

        # copy weights to new model
        func_model.set_weights(seq_model.get_weights())

        # run model
        output, spikes = func_model.predict(test_sequences)

        if has_global_average_pooling:
            # check test accuracy using average output over all timesteps
            predictions = np.argmax(output.mean(axis=1), axis=-1)
        else:
            # check test accuracy using output from only the last timestep
            predictions = np.argmax(output[:, -1], axis=-1)
        accuracy = np.equal(predictions, test_labels).mean()
        print(f"Test accuracy: {100 * accuracy:.2f}%")

        time = test_sequences.shape[1] * spike_layer.dt
        n_spikes = spikes * spike_layer.dt
        rates = np.sum(n_spikes, axis=1) / time

        print(
            f"Spike rate per neuron (Hz): min={np.min(rates):.2f} "
            f"mean={np.mean(rates):.2f} max={np.max(rates):.2f}"
        )

        # plot output
        for ii in range(4):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title(class_names[test_labels[ii]])
            plt.imshow(test_images[ii], cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Spikes per neuron per timestep")
            bin_edges = np.arange(int(np.max(n_spikes[ii])) + 2) - 0.5
            plt.hist(np.ravel(n_spikes[ii]), bins=bin_edges)
            x_ticks = plt.xticks()[0]
            plt.xticks(
                x_ticks[(np.abs(x_ticks - np.round(x_ticks)) < 1e-8) & (x_ticks > -1e-8)]
            )
            plt.xlabel("# of spikes")
            plt.ylabel("Frequency")

            plt.subplot(1, 3, 3)
            plt.title("Output predictions")
            plt.plot(
                np.arange(test_sequences.shape[1]) * spike_layer.dt,
                tf.nn.softmax(output[ii]),
            )
            plt.legend(class_names, loc="upper left")
            plt.xlabel("Time (s)")
            plt.ylabel("Probability")
            plt.ylim([-0.05, 1.05])

            plt.tight_layout()

    check_output(spiking_model, modify_dt=0.1)