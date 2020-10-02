import random, math

class LinearModel:
    def __init__(self, m=None, b=None):
        """
        Initializes the slope and y-intercept of the regression line with either default values
        or the values that have been passed as arguments.

        Parameters
        ---------
        m (float):
            an initial value for the slope
        b (float):
            an initial value for the y-intercept
        """
        self.m = m or random.random()
        self.b = b or random.random()

        # after each iteration, we'll add a dictionary containing the current values of to
        # the history object, e.g. {'m':11.5, 'b':-3.42, 'loss':.25, 'gradient':(-1.2, 3.5)}
        self.history = []

    def error(self, x=None, y=None):
        """
        Returns the error of the observation (x,y) passed, computed as
        self.

        Parameters
        ----------
        x (float):
            the x-value of the observation
        y (float):
            the y-value of the observation
        """
        return self.predict(x) - y

    def gradient(self, x, y):
        """
        Computes the gradient of the loss function at this observation, computed as

        (x * (mx+b - y), (mx+b - y))

        Parameters
        ----------
        x (float):
            the x-value of the observation where we want the gradient
        y (float):
            the y-value of the observation where we want the gradient

        Return
        ------
        A list containing ∂L/∂m, ∂L/∂b
        """
        return x * self.error(x,y), self.error(x,y)

    def loss(self, x, y):
        """
        Computes the loss, which is (mx+b - y)^2

        Parameters
        ----------
        x (float):
            the x-value of the observation
        y (float):
            the y-value of the observation

        Returns
        -------
        The loss, (prediction-actual)^2 = (mx+b - y)**2
        """
        return self.error(x, y)**2

    def predict(self, x):
        """
        Accepts an x-value and returns the prediction, m*x + b

        Parameters
        ----------
        x (float):
            the x-value whose y-value we want to predict

        Returns
        -------
        the prediction mx + b with the current values of m and b
        """
        return self.m*x + self.b

    def rmse(self, xlist, ylist):
        """
        Computes the rmse for the data passed.

        Parameters
        ----------
        xlist (array-like):
            an array containing the x-values
        ylist (array-like):
            an array c
        """
        return math.sqrt(sum(list(self.loss(x,y) for x,y in zip(xlist, ylist)))) / len(xlist)



    def step(self, m_step, b_step):
        """
        Accepts a step for m and b, and adds those to their counterparts.

        Parameters
        ----------
        m_step (float):
            the amount to step along the m-axis
        b_step (float):
            the amount to step along the b-axis

        Returns
        -------
        None
        """
        self.m += m_step
        self.b += b_step

    def update_history(self, x, y, epoch=0, message=None):
        self.history.append({'m':self.m,
                            'b':self.b,
                            'observation':(x,y),
                            'prediction':self.predict(x),
                            'error':self.error(x,y),
                            'loss':self.loss(x,y),
                            'epoch':epoch,
                            'message':message})

    def __str__(self):
        """
        Returns a string containing the equation of the regression line.

        Returns
        -------
        A formatted linear equation.
        """
        return 'y = {:.3f}x + {:.3f}'.format(self.m, self.b)
