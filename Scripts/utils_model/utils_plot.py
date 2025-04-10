import numpy as np
import scipy
import matplotlib.pyplot as plt


def plot_two_spectrum(true, pred, start, stop, marks=False, fill=True, rescale=1, line_width=4, fontsize=30):
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    x = np.linspace(start, stop, len(true))

    peaks_true, _ = scipy.signal.find_peaks(true)
    peaks_pred, _ = scipy.signal.find_peaks(pred)

    # Rescale the intensity of the Fingerprint Region
    true[:750] = true[:750] * rescale
    pred[:750] = pred[:750] * rescale

    # Draw Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)

    if fill:
        ax.fill_between(x, y1=true, y2=0, label='DFT-Calculated Raman Spectrum', alpha=0.5, color=mycolors[1],
                        linewidth=line_width/3.75)
        ax.fill_between(x, y1=pred, y2=0, label='Predicted Raman Spectrum', alpha=0.5, color=mycolors[0],
                        linewidth=line_width/3.75)
    else:
        ax.plot(x, true, label='DFT-Calculated Spectrum', color=mycolors[1], linewidth=line_width / 3.75)
        ax.plot(x, pred, label='Predicted Spectrum', color=mycolors[0], linewidth=line_width/ 3.75)

    # #Croci e punti sui picchi
    if marks:
        plt.scatter(x[peaks_true], true[peaks_true], color='blue',
                    marker='o', s=25, label='True Local Maxima')
        plt.scatter(x[peaks_pred], pred[peaks_pred], color='red',
                    marker='x', s=25, label='Predicted Local Maxima')

    # Decorations
    # ax.set_title('Raman Spectrum (DFT-Calculated vs Predicted)', fontsize=18/3.75)
    ax.set_xlabel('Raman shift ($cm^{-1}$)', fontsize=fontsize/3.75)
    ax.set_ylabel('Intensity (a.u.)', fontsize=fontsize/3.75)
    ax.legend(loc='best', fontsize=fontsize/3.75)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    plt.xticks(fontsize=fontsize/3.75, horizontalalignment='center')
    plt.yticks(fontsize=fontsize/3.75)
    plt.xlim(start, stop)
    plt.ylim(bottom=0)

    # # Thickness of the plot corner lines
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)

    # Draw Tick lines
    # for y in np.arange(0, max(max(true), max(pred)), step=0.1):
    #     plt.hlines(y, xmin=start, xmax=stop, colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


def plot_three_spectrum(true, pred1, pred2, start, stop, marks=False, fill=True, rescale=1, line_width=4):
    """

    """
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    x = np.linspace(start, stop, len(true))

    peaks_true, _ = scipy.signal.find_peaks(true)
    peaks_pred1, _ = scipy.signal.find_peaks(pred1)
    peaks_pred2, _ = scipy.signal.find_peaks(pred2)

    # Rescale the intesity of the Fingerprint Region
    true[:750] = true[:750] * rescale
    pred1[:750] = pred1[:750] * rescale
    pred2[:750] = pred2[:750] * rescale

    # Draw Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)

    if fill:
        ax.fill_between(x, y1=true, y2=0, label='DFT-Calculated Spectrum', alpha=0.5, color=mycolors[1],
                        linewidth=line_width/3.75)
        ax.fill_between(x, y1=pred1, y2=0, label='Predicted First Spectrum', alpha=0.5, color=mycolors[0],
                        linewidth=line_width/3.75)
        ax.fill_between(x, y1=pred2, y2=0, label='Predicted Second Spectrum', alpha=0.5, color=mycolors[2],
                        linewidth=line_width/3.75)
    else:
        ax.plot(x, true, label='DFT-Calculated Spectrum', color=mycolors[1], linewidth=line_width / 3.75)
        ax.plot(x, pred1, label='Predicted First Spectrum', color=mycolors[0], linewidth=line_width/ 3.75)
        ax.plot(x, pred2, label='Predicted Second Spectrum', color=mycolors[2], linewidth=line_width/ 3.75)

    # Croci e punti sui picchi
    if marks:
        plt.scatter(x[peaks_true], true[peaks_true], color='blue',
                    marker='o', s=25, label='True Local Maxima')
        plt.scatter(x[peaks_pred1], pred1[peaks_pred1], color='red',
                    marker='x', s=25, label='Predicted Local Maxima')
        plt.scatter(x[peaks_pred2], pred1[peaks_pred2], color='red',
                    marker='x', s=25, label='Predicted Local Maxima')
    # Decorations
    ax.set_title('Raman Spectrum (DFT-Calculated vs Predicted)', fontsize=18/3.75)
    ax.set_xlabel('Raman shift ($cm^{-1}$)', fontsize=25/3.75)
    ax.set_ylabel('Intensity (a.u.)', fontsize=25/3.75)
    ax.legend(loc='best', fontsize=18/3.75)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    plt.xticks(fontsize=25/3.75, horizontalalignment='center')
    plt.yticks(fontsize=25/3.75)
    plt.xlim(start, stop)
    plt.ylim(bottom=0)

    # # Thickness of the plot corner lines
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)

    # Draw Tick lines
    # for y in np.arange(0, max(max(true), max(pred)), step=0.1):
    #     plt.hlines(y, xmin=start, xmax=stop, colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


def plot_bar_f1(df):
    bin = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    plt.figure(figsize=(10, 6))
    plt.hist(df['f1'], bins=bin, color='skyblue', edgecolor='black')

    plt.xlabel('F1 Score')
    plt.ylabel('Numero di Molecole')
    plt.text(df.f1.mean() + 0.01, max(np.histogram(df['f1'], bins=bin)[0]) - 1, f'Mean: {df.f1.mean():.2f}', color='red')

    plt.tight_layout()
    plt.show()
