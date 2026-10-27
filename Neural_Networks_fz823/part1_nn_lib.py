import numpy as np
import pickle


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Array containing element-wise transformation of the
                inputs using the sigmoid function, of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Cache the output, not the input: the derivative is written in terms
        # of it, sigma'(x) = sigma(x) * (1 - sigma(x)).
        self._cache_current = 1 / (1 + np.exp(-x))
        return self._cache_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs a backward pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Chain rule, element-wise: dL/dx = dL/dz * dz/dx.
        s = self._cache_current
        return grad_z * s * (1 - s)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Array containing element-wise transformation of the
                inputs using the Relu function, of shape (batch_size, n_out).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Cache the input so backward knows which units were active (x > 0).
        self._cache_current = x
        return np.maximum(0, x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # relu'(x) is 1 where x > 0 and 0 elsewhere: the gradient only flows
        # back through units that were active in the forward pass.
        return grad_z * (self._cache_current > 0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Applies a linear transformation to the input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Uses NumPy arrays to represent the learnable parameters of the layer,
        initialized in a sensible manner (hint: you can use the provided 
        xavier_init function). Attributes _W, _b refer to the weights matrix 
        and bias respectively.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Xavier keeps the output variance similar to the input variance, so
        # signals neither explode nor vanish as layers stack. Bias starts at 0.
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros(n_out)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs a forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute the gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # The input X is needed later for dL/dW.
        self._cache_current = x
        # (batch, n_in) @ (n_in, n_out) -> (batch, n_out); b broadcasts to rows.
        return x @ self._W + self._b

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).
        
        Logs computed gradients in relevant attributes (`_grad_W_current` and
        `_grad_b_current`).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Z = XW + b gives:
        #   dL/dW = X^T @ dL/dZ   (n_in, batch) @ (batch, n_out) -> (n_in, n_out)
        #   dL/db = dL/dZ summed over the batch (every row shares the same b)
        #   dL/dX = dL/dZ @ W^T   (batch, n_out) @ (n_out, n_in) -> (batch, n_in)
        self._grad_W_current = self._cache_current.T @ grad_z
        self._grad_b_current = grad_z.sum(axis=0)
        return grad_z @ self._W.T

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with the given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Step each parameter against its gradient (downhill on the loss).
        self._W = self._W - learning_rate * self._grad_W_current
        self._b = self._b - learning_rate * self._grad_b_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork:
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Initialises instances of the LinearLayer class
        and activation classes, as specified by the given arguments.
        Saves the layer instances in the `_layers` variable.

        Arguments:
            - input_dim {int} -- The number of features in the input data
                (excluding the batch dimension).
            - neurons {list} -- Number of neurons in the output of each linear layer, 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer (e.g. "sigmoid", "relu",
                or "identity" for linear passthrough / no activation).
                The length of `activations` should match the length of `neurons`.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        self._layers = []

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        activation_classes = {"relu": ReluLayer, "sigmoid": SigmoidLayer}
        n_in = input_dim
        for n_out, activation in zip(neurons, activations):
            self._layers.append(LinearLayer(n_in, n_out))
            # "identity" is a linear passthrough, so it adds no layer at all.
            if activation in activation_classes:
                self._layers.append(activation_classes[activation]())
            elif activation != "identity":
                raise ValueError(f"Unknown activation: {activation}")
            n_in = n_out  # this layer's output feeds the next layer's input
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)

        Notes:
            The instances of the LinearLayer classes created in the constructor
            should automatically handle any storage of data needed to compute 
            gradients.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            x = layer(x)
        return x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).

        Notes:
            The instances of the LinearLayer classes created in the constructor
            should automatically handle any storage of computed gradients.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Backpropagation: walk the layers in reverse. Each one turns
        # dL/d(its output) into dL/d(its input) for the layer before it.
        for layer in reversed(self._layers):
            grad_z = layer.backward(grad_z)
        return grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with the given learning rate on the
        parameters of all layers using currently stored gradients and the 
        learning rate provided as argument.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Activation layers have no parameters; they inherit Layer's no-op.
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer:
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun="mse",
        shuffle_flag=True,
        log_freq=50,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.

        Keyword Arguments:
            shuffle_flag {bool} -- If True, training data is shuffled. (default:
                {True})
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy (default: {"mse"})
            log_freq {int} -- Logs training loss every multiple of log_freq
                epochs (default: {50})
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.shuffle_flag = shuffle_flag
        self.loss_fun = loss_fun
        self.log_freq = log_freq

        if self.loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif self.loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            raise ValueError("Unknown loss function")

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        assert len(input_dataset) == len(target_dataset)
        shuffled_idx = np.arange(0, len(input_dataset))
        np.random.shuffle(shuffled_idx)
        return input_dataset[shuffled_idx], target_dataset[shuffled_idx]

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        assert len(input_dataset) == len(target_dataset)

        nb_batches = int(len(input_dataset) / self.batch_size)
        for epoch in range(self.nb_epoch):
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(
                    input_dataset, target_dataset
                )
            for i in range(0, len(input_dataset), self.batch_size):
                input_batch = input_dataset[i:i+self.batch_size]
                target_batch = target_dataset[i:i+self.batch_size]

                pred = self.network(input_batch)
                loss = self._loss_layer(pred, target_batch)
                grad_z = self._loss_layer.backward()
                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)

            if epoch % self.log_freq == 0:
                print(
                    "Epoch {}: training loss (last batch) = {}".format(
                        epoch, loss
                    )
                )

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns 
        scalar value.
        
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).

        Returns:
            - a scalar value -- the loss
        """
        assert len(input_dataset) == len(target_dataset)
        pred = self.network(input_dataset)
        return self._loss_layer(pred, target_dataset)


class Preprocessor:
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Constructor of the Preprocessor.
        
        Computes and stores data normalization parameters according 
        to the provided dataset.
        The constructor should not modify the dataset.

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Per-column (axis=0) statistics, so every feature gets its own scale.
        self._min = data.min(axis=0)
        data_range = data.max(axis=0) - self._min
        # A constant column would divide by 0; range 1 maps it to 0 instead.
        self._range = np.where(data_range == 0, 1, data_range)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.
        Performs min-max scaling such that the data is scaled to lie 
        in the interval [0, 1].

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return (data - self._min) / self._range

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Exact inverse of apply: undo the division, then undo the shift.
        return data * self._range + self._min

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print(f"Train loss = {trainer.eval_loss(x_train_pre, y_train)}")
    print(f"Validation loss = {trainer.eval_loss(x_val_pre, y_val)}")

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print(f"Validation accuracy: {accuracy}")


if __name__ == "__main__":
    example_main()
