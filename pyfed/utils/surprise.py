import numpy as np

EPS = 1e-8
NORM = False


def prep_bins_accurate(ps, qs, edges=None, alpha=0.01, value="probs", smooth_q=True):
    """
    Prepare bins. Includes removing end bins with zero counts and add-one smoothing.
    TODO:
        1) Compare performance of removing zero bins to leaving them (add-one smoothing)
            -i) Speed (array processing given same number of bins)
            -ii) Accuracy (change in scores for all but dead units)
            -iii) How big do N and M have to be in order to have no significant difference?
        2) Add-one smoothing: p only?
            -i) Issue: 1 count for p, 0 for q --> large Opp! Reduces surprise *increase*
            -ii) Solution: Either remove zero bins and add-one to p, or leave zero bins and add-one to p AND q.
    :param ps: train dist bin counts.
    :param qs: test dist bin counts.
    :param edges: bin edges.
    :param alpha: smoothing "pseudocount". 0 corresponds to no smoothing, while 1 corresponds to add-one smoothing.
    :param value: value to return.
    :return: array: counts, probs or densities.
    """
    valid_values = ["counts", "probs", "densities"]
    if value not in valid_values:
        raise ValueError("Invalid value selected {0}.\nChoose one of {1}".format(value, valid_values))

    # get total counts (assuming they are the same for each feature)
    n_ps = ps[0].sum()
    n_qs = qs[0].sum()

    # remove bins where both obs and exp counts are zero. New name to avoid unexpected behaviour from mutable ps,qs
    nonzero_inds = (ps + qs) != 0
    ps_ = [p[nonz_is] for p, nonz_is in zip(ps, nonzero_inds)]
    qs_ = [q[nonz_is] for q, nonz_is in zip(qs, nonzero_inds)]

    # add-one smoothing to prevent extreme values (p only?)
    ks = [len(p) for p in ps_]  # num non-empty bins for each feature
    ps_ = [p + alpha for p in ps_]
    n_ps = np.array([n_ps + k * alpha for k in ks])  # update total counts
    if smooth_q:
        qs_ = [q + alpha for q in qs_]
        n_qs = np.array([n_qs + k * alpha for k in ks])  # update total counts
    else:
        n_qs = np.array([n_qs for _ in ks])

    # calculate desired value
    if value == "counts":
        return ps_, qs_, n_ps, n_qs

    if value == "probs":
        ps_ = [p / float(n_p) for p, n_p in zip(ps_, n_ps)]
        qs_ = [q / float(n_q) for q, n_q in zip(qs_, n_qs)]
        return ps_, qs_, n_ps, n_qs

    if value == "densities":
        if edges is None:
            raise ValueError("Need bin edges (widths) to calculate densities!")
        hs = edges[:, 1:] - edges[:, :-1]  # bin widths
        hs = [h[nonz_is] for h, nonz_is in zip(hs, nonzero_inds)]  # bin widths for non-zero bins
        ps_ = [p / h / float(n_p) for p, h, n_p in zip(ps_, hs, n_ps)]
        qs_ = [q / h / float(n_q) for q, h, n_q in zip(qs_, hs, n_qs)]
        return ps_, qs_, n_ps, n_qs


def prep_bins_fast(p_cs, q_cs, edges=None, alpha=0.01, value="probs", smooth_q=True):
    valid_values = ["counts", "probs", "densities"]
    if value not in valid_values:
        raise ValueError("Invalid value selected {0}.\nChoose one of {1}".format(value, valid_values))

    p_cs_ = p_cs.copy()  # mutable object
    q_cs_ = q_cs.copy()  # mutable object
    nonzero_inds = (p_cs + q_cs) != 0
    n_bins = nonzero_inds.sum(1).reshape(-1, 1)

    # smoothing using "pseudocount" alpha. Prevents extreme values (p only for surprise). alpha=0 means no smoothing.
    p_cs_ += alpha
    if smooth_q:
        q_cs_ += alpha

    # get total counts (assuming they are the same for each feature)
    n_ps = p_cs_.sum(1).reshape(-1, 1)
    n_qs = q_cs_.sum(1).reshape(-1, 1)

    # calculate desired value
    if value == "counts":
        return p_cs_, q_cs_, n_ps, n_qs, n_bins

    if value == "probs":
        p_cs_ = p_cs_ / n_ps
        q_cs_ = q_cs_ / n_qs
        return p_cs_, q_cs_, n_ps, n_qs, n_bins

    if value == "densities":
        if edges is None:
            raise ValueError("Need bin edges (widths) to calculate densities!")
        hs = edges[:, 1:] - edges[:, :-1] + EPS  # non-zero bin widths
        p_cs_ = p_cs_ / hs / n_ps
        q_cs_ = q_cs / hs / n_qs
        return p_cs_, q_cs_, n_ps, n_qs, n_bins


def bin_variance(ps, n_qs, norm=NORM):
    """
    Calculate the Gaussian approx for the confidence interval under the null hypothesis that the q_i's are
    obtained as q_i = x_i/n by sampling n times from the multinomial with probs p = (p_1, ..., p_k) to obtain
    counts(x_1, ..., x_k).
    :param ps: (f x r) array of probs p = (p_1, ..., p_k), where f is the num of features of k is the num of bins.
    :param n_qs: array containing the number of q "samples" for each feature (usually the same...)
    :param norm: "normalised" surprise score is being used (dividing score by h_p)
    :return: array of variances that define confidence intervals around the surprise score.
    """
    vs = []
    for p_is, n in zip(ps, n_qs):
        if len(p_is) == 1:
            # single bin with prob = 1. for both p and q
            vs.append(EPS)
            continue
        w = - np.log(p_is + EPS)  # surprise
        var_di = p_is * (1. - p_is) / n
        cov_di_dj = - np.outer(p_is, p_is) / n
        cov_d = cov_di_dj - np.diag(np.diag(cov_di_dj)) + np.diag(var_di)
        var_stat = w.T.dot(cov_d).dot(w)
        if norm:
            o_pp = (p_is.dot(w) + EPS)
            var_stat /= ((o_pp ** 2) + EPS)
        vs.append(var_stat)

    return np.array(vs)


def get_bin_entropies(p_cs, q_cs, fast=True, alpha=0.01):
    if fast:
        # Prep bins: remove bins where p_count AND q_count = 0, smooth p and q, counts  --> probabilities
        ps, qs, _, n_qs, n_bins = prep_bins_fast(p_cs, q_cs, alpha=alpha)

        # Change log base = num bins (put h_p and h_q in range [0,1])
        log_base_change_divisor = np.log(n_bins)
        single_bin_inds = (log_base_change_divisor == 0.)  # single bin with prob = 1. for both p and q
        log_base_change_divisor[single_bin_inds] = 1.

        # Calculate entropies
        h_ps = -np.sum(ps * np.log(ps + EPS) / log_base_change_divisor, 1)
        h_qs = -np.sum(qs * np.log(qs + EPS) / log_base_change_divisor, 1)
        h_q_ps = -np.sum(qs * np.log(ps + EPS) / log_base_change_divisor, 1)
        h_p_qs = -np.sum(ps * np.log(qs + EPS) / log_base_change_divisor, 1)

        # Set entropy = exactly zero for units/features with a single bin prob = 1. for both p and q
        h_ps[single_bin_inds[:, 0]] = 0.
        h_qs[single_bin_inds[:, 0]] = 0.
        h_q_ps[single_bin_inds[:, 0]] = 0.
        h_p_qs[single_bin_inds[:, 0]] = 0.

    else:  # cannot do array-based operations as each feature has a diff num of bins
        # Prep bins: remove bins where p_count AND q_count = 0, smooth p (length of code), counts  --> probabilities
        ps, qs, _, n_qs = prep_bins_accurate(p_cs, q_cs, alpha=alpha)
        n_bins = np.array([len(p) for p in ps])

        # Calculate entropies
        h_ps = []
        h_qs = []
        h_q_ps = []
        h_p_qs = []
        for i, (p_is, q_is, k) in enumerate(zip(ps, qs, n_bins)):
            if len(p_is) == 1:  # single bin with prob = 1. for both p and q. Zero entropy/randomness!
                h_ps.append(0.), h_qs.append(0.), h_q_ps.append(0.), h_p_qs.append(0.),
                continue
            h_p = -np.sum(p_is * np.log(p_is + EPS) / np.log(k))
            h_q = -np.sum(q_is * np.log(q_is + EPS) / np.log(k))
            h_q_p = -np.sum(q_is * np.log(p_is + EPS) / np.log(k))
            h_p_q = -np.sum(p_is * np.log(q_is + EPS) / np.log(k))
            h_ps.append(h_p), h_qs.append(h_q), h_q_ps.append(h_q_p), h_p_qs.append(h_p_q)
        h_ps, h_qs, h_q_ps, h_p_qs = np.array(h_ps), np.array(h_qs), np.array(h_q_ps), np.array(h_p_qs)

    return h_ps, h_qs, h_q_ps, h_p_qs, ps, qs, n_qs


def surprise_bins(p_cs, q_cs, score_type="KL_Q_P", fast=True, alpha=0.01, bin_edges=None):
    """
    Calculate desired surprise score given bin counts.
    :param p_cs: bin counts from P.
    :param q_cs: bin counts from Q.
    :param score_type: desired score type (str).
    :param fast: sacrifice some accuracy for speed (bool). Depends on method for bins that are empty for both p and q.
    :param alpha: smoothing 'pseudocount' (float).
    :return: array of scores.
    Note: counts for *both* p and q are currently smoothed using the same alpha, regardless of which defines the
    code length (log(p) / log(q)). This makes calculations easier and entropy calculations stable. However, there may
    be reason not to smooth p and/or q. Would involve get_bin_entropies(q_cs, p_cs).
    TODO:
        1) Can we reduce computation by storing ps (probs) and h_p? Definitely if we can use "empirical" H(q,p) --
        see surprise_MoGs fn.
    """
    if score_type == "JS":  # Jensen–Shannon divergence, M is the average of distributions P and Q.
        m_cs = 0.5 * (p_cs + q_cs)
        h_p, _, _, h_p_m, _, _, _ = get_bin_entropies(p_cs, m_cs, fast=fast, alpha=alpha)
        h_q, _, _, h_q_m, _, _, _ = get_bin_entropies(q_cs, m_cs, fast=fast, alpha=alpha)  # slight redundancy in calcs
        return 0.5 * (h_p_m - h_p) + 0.5 * (h_q_m - h_q)

    h_p, h_q, h_q_p, h_p_q, ps, qs, n_qs = get_bin_entropies(p_cs, q_cs, fast=fast, alpha=alpha)

    if score_type == "SI":
        return h_q_p - h_p
    elif score_type == "SI_norm":
        return (h_q_p - h_p) / (np.abs(h_p) + EPS)
    elif score_type == "SI_Z":
        sampling_std = np.sqrt(np.abs(bin_variance(ps, n_qs)) + EPS)
        return (h_q_p - h_p) / (sampling_std + EPS)
    elif score_type == "KL_Q_P":  # KL(Q||P)
        return h_q_p - h_q
    elif score_type == "KL_P_Q":  # KL(P||Q)
        return h_p_q - h_p
    elif score_type == "PSI":  # Population stability index (Symmetric KL divergence)
        return (h_p_q - h_p) + (h_q_p - h_q)
    elif score_type == "EMD":
        raise NotImplementedError("Coming soon from https://github.com/wmayner/pyemd")
        #  TODO:
        #   1) bin edges need to match counts when zero-bins have been removed.
        #   2) Unit testing (speed + accuracy).
        # if bin_edges is None:
        #     raise ValueError("Bin edges are required to compute the distance matrix for the EMD(P,Q).")
        # emds = np.zeros(len(ps))
        # bin_locations = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)                        # centre of bins
        # for i, (unit_ps, unit_qs, unit_bin_locations) in enumerate(zip(ps, qs, bin_locations)):
        #     distance_matrix = euclidean_pairwise_distance_matrix(unit_bin_locations)            # k x k matrix
        #     emds[i] = emd(unit_ps, unit_qs, distance_matrix)
        # return np.array(emds)
    else:
        raise ValueError("Invalid surprise score choice {0} for bins.".format(score_type))
