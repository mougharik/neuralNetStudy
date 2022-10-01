# Neural Network Comparative Study
Comparing various architectures (fully-connected and multiple convolutional variants) on the USPS dataset for the optical recognition task. Currently, four different weight initialization presets are available for training the models: <i>effective</i>, <i>too slow</i>, <i>too fast</i>, and <i>default</i>. In addition to weight initialization schemes, various learning rates have been tested and categorized in a similar manner (effective, too slow, too fast). Please refer to the <i>materials</i> folder for a more detailed write-up.

## Example Usage
    python3 study.py --net 1    # fully-connected net
    python3 study.py --net 2    # locally-connected CNN
    python3 study.py --net 3    # fully-connected CNN
<br>

    python3 study.py --net 1 --init 1   # effective learning
    python3 study.py --net 1 --init 2   # fast learning
    python3 study.py --net 1 --init 3   # slow learning
