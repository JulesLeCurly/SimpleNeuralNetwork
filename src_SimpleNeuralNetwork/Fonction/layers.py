
class Dense:
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        name=None
    ):
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

class InputLayer:
    def __init__(
        self,
        units,
        name=None
    ):
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.name = name