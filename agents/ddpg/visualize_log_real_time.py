import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

def stats(x,window_len=11):
    df_x = pd.Series(x)
    rolling_mean = df_x.rolling(window_len).mean().values
    rolling_std = df_x.rolling(window_len).std().values
    return rolling_mean, rolling_std

def update(i):
    print('update')
    try:
        with open(args.filename, 'r') as f:
            data = json.load(f)
        if 'episode' not in data:
            raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
        episodes = data['episode']

        # Get value keys. The x axis is shared and is the number of episodes.
        keys = sorted(list(set(data.keys()).difference(set(['episode']))))
        for idx, key in enumerate(keys):

            y = np.array(data[key])
            y_rolling_mean, y_rolling_std = stats(y)
            axarr[idx].cla()
            axarr[idx].fill_between(episodes, y_rolling_mean - y_rolling_std,
                             y_rolling_mean + y_rolling_std, alpha=0.25,
                             color="r")
            axarr[idx].plot(episodes, y_rolling_mean, alpha=1, color='red', linewidth=1)

            axarr[idx].plot(episodes, y, alpha=0.5, color='black', linewidth=0.1)


            axarr[idx].set_ylabel(key)
        plt.xlabel('episodes')
        plt.tight_layout()
    except Exception as e:
        print(e)

def init():
    global axarr
    print('init')

    with open(args.filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if args.figsize is None:
        args.figsize = (15., 5. * len(keys))
    fig, axarr = plt.subplots(len(keys), sharex=True, figsize=args.figsize)
    return fig

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.


fig = init()
ani = animation.FuncAnimation(fig, func=update, interval=1000)
plt.show()

