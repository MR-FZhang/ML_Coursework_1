import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn
from sklearn import preprocessing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

class Regressor:

    # Defaults chosen on a held-out 20% split: validation RMSE plateaus by
    # ~150 epochs, and wider/deeper networks only overfit.
    def __init__(self, x, nb_epoch = 150, hidden_layers = (64, 64),
                 learning_rate = 1e-3, batch_size = 64):
        """ 
        Initialise the regression model.

	You can add any input parameters you need.
        Just remember to set them with default values, so the LabTS tests still work.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - hidden_layers {tuple} -- number of neurons in each hidden layer.
            - learning_rate {float} -- step size for the Adam optimiser.
            - batch_size {int} -- number of rows per gradient step.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Hidden layers use ReLU; the output layer stays linear because this
        # is regression (predict any real value), not classification.
        layers = []
        n_in = self.input_size
        for n_out in hidden_layers:
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        layers.append(nn.Linear(n_in, self.output_size))
        self.model = nn.Sequential(*layers)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        # Build a copy of the dataframes
        given_y = isinstance(y, pd.DataFrame)
        X = x.copy()
        if given_y:
            Y = y.copy()

        # Set parameters for preprocessor
        if training :
            self.preproc_text = {}
            pre_X = X.select_dtypes(include=['float'])
            self.preproc_fill = pd.DataFrame( [pre_X.median().to_list(), \
                                       pre_X.min().to_list(), \
                                       pre_X.max().to_list()],\
                                      columns = pre_X.columns.to_list())
            if given_y:
                self.preproc_fill_out = [Y.iloc[:, 0].median(), \
                                 Y.iloc[:, 0].min(), \
                                 Y.iloc[:, 0].max()]

        # Replace missing values in X
        X.fillna(self.preproc_fill.iloc[0], inplace=True) 

        # Replace textual values in X
        for column in X.select_dtypes(include=['object']).columns:
            lb = preprocessing.LabelBinarizer()
            if training :
                lb.fit(X[column])
                self.preproc_text[column] = lb.classes_
            else:
                lb.classes_ = self.preproc_text[column]
            values = lb.transform(X[column])
            X = X.loc[:, X.columns != column] # Remove current column
            for new_col in range (len(lb.classes_)): # Add new ones
                X.insert(len(X.columns), column + "_" + str(new_col), values[:, new_col])

        # Normalise
        for column in X.select_dtypes(include=['float']).columns:
            X[column] = X[column].subtract(self.preproc_fill[column].iloc[1]) \
                .div(self.preproc_fill[column].iloc[2] - self.preproc_fill[column].iloc[1])
        if given_y:
            if (Y.iloc[:, 0] > 1).any():
                Y.iloc[:,0] = Y.iloc[:,0].subtract(self.preproc_fill_out[1]) \
                    .div(self.preproc_fill_out[2] - self.preproc_fill_out[1])

        return torch.tensor(X.values.astype('float'), dtype = torch.float), \
               torch.tensor(Y.values.astype('float'), dtype = torch.float) if given_y \
                                                                           else None


    def _revert_preprocessor(self, y):
        """ 
        Reverse the normalisation of the targets.
          
        Arguments:
            y {torch.tensor} -- Raw target array of shape (batch_size, 1).

        Returns:
            {np.ndarray} -- Unnormalised target values (batch_size, 1).
        """
        y = y * (self.preproc_fill_out[2] - self.preproc_fill_out[1]) + self.preproc_fill_out[1]
        return y.detach().numpy()


    def fit(self, x, y):
        """
        Regressor training function

        Performs the following steps number-of-epochs times:
            - Performs forward pass through the model given the input.
            - Computes the loss based on this forward pass.
            - Performs backwards pass to compute gradients of loss with 
              respect to parameters of the model.
            - Performs one step of gradient descent on the model parameters.
        You are free to implement any additional steps to improve 
              learning (batch-learning, shuffling,...).

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        optimiser = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()

        for _ in range(self.nb_epoch):
            # Fresh shuffle each epoch so the mini-batches differ every pass.
            permutation = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                batch = permutation[i:i + self.batch_size]
                optimiser.zero_grad()   # PyTorch accumulates grads; reset them
                loss = loss_fn(self.model(X[batch]), Y[batch])  # forward + loss
                loss.backward()         # backprop: fills .grad of every weight
                optimiser.step()        # gradient-descent update of the weights
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x using the regressor model.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).
        
         Hint:
            You can use the provided _preprocessor and _revert_preprocessor 
            functions.
        
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.model.eval()
        with torch.no_grad():  # inference only: skip building the autograd graph
            # The network predicts in the scaled [0, 1] space; map back to dollars.
            return self._revert_preprocessor(self.model(X))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def evaluate(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.
        This should call predict(), which itselt should call the preprocessor.
        Returns RMSE for the regression model on the given dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- True target values. Array of shape (batch_size, 1).

        Returns:
            {float} -- Returns RMSE on this dataset. RMSE(f(x), y)

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # RMSE = sqrt(mean((y - y_pred)^2)), in the same units as the target.
        return float(root_mean_squared_error(y, self.predict(x)))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def main():
    """
    The main function to train, save and evaluate the regression model.
    This function is not called by LabTS, so feel free to change it.
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Hold out 20% the model never trains on: its RMSE estimates performance
    # on unseen data, which is what the hidden LabTS test measures.
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size = 0.2, random_state = 42)

    # Training
    torch.manual_seed(0)  # reproducible weight init and shuffling
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    print(f"\nTrain RMSE: {regressor.evaluate(x_train, y_train)}")
    print(f"Validation RMSE: {regressor.evaluate(x_val, y_val)}\n")


if __name__ == "__main__":
    # Run main() via an import so the pickle records the class as
    # part2_house_value_regression.Regressor rather than __main__.Regressor;
    # LabTS imports this module, so it can then always unpickle the model.
    import part2_house_value_regression
    part2_house_value_regression.main()

