# Generalized-LSTM

This small project enables the generalization of Long Short-Term Memory (LSTM) networks to other operations such as convolution. It proves useful when a custom operation is required and/or it is not feasible to use a fully-connected layer, allowing for a reduction in the number of operations.

## Overview

LSTM networks are powerful for sequence modeling tasks, but sometimes it's necessary to extend their capabilities to other operations like convolution. This project provides a solution for such scenarios, enabling the integration of LSTM functionalities into a broader range of operations.

## Key Features

- **Generalization**: Extend LSTM capabilities to operations beyond sequence modeling.
- **Custom Operations**: Use this project when a custom operation is needed.
- **Reduced Operations**: Avoid the necessity of using a fully-connected layer, reducing the overall number of operations.

## How to Use

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/adriaciurana/generalized-LSTM.git
   ```

2. Follow the installation instructions in the provided documentation.

3. Integrate the project into your existing codebase.

4. Customize the operation as needed for your specific use case.

5. Enjoy the flexibility of using LSTM functionalities in a broader context.

## Requirements

- Python 3.x
- No extra dependencies

## Example

```python
from generalized_lstm import Conv2dLSTM

# Initialize the Conv2dLSTM
layer = Conv2dLSTM(3, 32, 3, num_layers=2).eval().to(device)

# Perform custom operations using the Conv2dLSTM object
result = layer(input_data)

# Continue with your specific application logic
```

## Contributions

Contributions are welcome! If you have ideas for improvements or new features, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).