from math import log


def calc_loss(log_p, log_det, image_size, n_bins):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + log_det + log_p
    log2 = log(2)
    return (
        (-loss / (log2 * n_pixel)).mean(),
        (log_p / (log2 * n_pixel)).mean(),
        (log_det / (log2 * n_pixel)).mean(),
    )
