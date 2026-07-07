"""
data_loading.py
----------------

Generalized data-loading module for the VAE-nlme pipeline (additive,
does NOT modify functions_theo.py / functions_neonates.py).

Both loaders below mirror the exact return-tuple contract of
functions_theo.load_data:

    (data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
     covariate_names, n_cov)

i.e. the original 7-tuple from functions_theo.load_data, PLUS two extra
trailing elements (covariate_names: list[str], n_cov: int) that the
original loaders don't return (because they hardcode covariate count/
identity for theophylline / neonates). Downstream code (Main/tacrolimus.py,
Main/paclitaxel.py) unpacks all 9 values.

Shapes (nbatch = number of subjects, T = max number of timepoints):
    data            : [nbatch, T, 3 + n_cov]   dim2 = [time, conc, dose, cov_1..cov_n]
    data_in         : [nbatch, T, 2]           standardized [time, conc] fed to Encoder
    lengths         : [nbatch]                 int, true number of observed timepoints per subject
    dose            : [nbatch]                 per-subject scalar dose (matches functions_theo's
                                                 `dose = data[:,0,2]` convention for a single dose
                                                 administered once per subject / used by a closed-form
                                                 decoder that broadcasts it across all timepoints)
    weight_pop      : scalar                   population mean of the first ("primary") covariate,
                                                 kept for parity with functions_theo (used there as
                                                 the log-covariate reference value in initalize_C)
    covariates      : [nbatch, n_cov]           raw (unstandardized) covariates
    covariates_in   : [nbatch, n_cov]           standardized covariates fed to Encoder
    covariate_names : list[str]                 length n_cov
    n_cov           : int

NOTE on the [time, concentration, dose, cov...] layout: functions_theo.load_data
assumes dim2 = [time, conc, dose, cov_1, ...] (it reads `dose = data[:,0,2]`,
`covariates = data[:,0,3:]`). We mirror that exactly: index 0 = time,
index 1 = concentration/DV, index 2 = dose, indices 3: = covariates.
"""

#########################################################
# Import
#########################################################
import csv
import io
import os

import numpy as np
import torch


#########################################################
# Helpers
#########################################################
def _sniff_delimiter(path, default=None):
    """Sniff whether a text table is comma- or whitespace-delimited."""
    with open(path, 'r') as f:
        first_line = f.readline()
    if default is not None:
        return default
    if ',' in first_line:
        return ','
    # whitespace-delimited (possibly multiple spaces/tabs) -- NONMEM-like .tab
    return None  # signals "split on any whitespace"


def _read_table(path, sep=None):
    """
    Read a NONMEM-like table into (header: list[str], rows: list[list[str]]).
    `sep`:
        None  -> auto-sniff (comma if present in header line, else whitespace)
        ','   -> comma-delimited
        any other string -> used as literal delimiter
    """
    delim = _sniff_delimiter(path) if sep is None else sep

    with open(path, 'r') as f:
        lines = [ln.rstrip('\n').rstrip('\r') for ln in f if ln.strip() != '']

    if delim == ',':
        reader = csv.reader(lines)
        rows = [row for row in reader]
    elif delim is None:
        rows = [ln.split() for ln in lines]
    else:
        rows = [ln.split(delim) for ln in lines]

    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


#########################################################
# Loader 1: NONMEM-like long format
#########################################################
def load_nonmem_like(path, id_col, time_col, dv_col, dose_col, covariate_cols=None, sep=None):
    """
    Load a NONMEM-like long-format table: one row per observation, with
    subject id + time + DV (concentration) + dose + covariate columns
    (covariates assumed constant within subject -- i.e. baseline covariates,
    as in theophylline_data.tab).

    Parameters
    ----------
    path : str
        Path to the table (space- or comma-delimited, with a header row).
    id_col, time_col, dv_col, dose_col : str
        Column names (from the header row) for subject id, time, DV
        (concentration), and dose respectively.
    covariate_cols : list[str] or None
        Column names to treat as covariates. If None, auto-detected as all
        columns other than id_col/time_col/dv_col/dose_col, preserving header
        order.
    sep : str or None
        Delimiter override; None auto-sniffs comma vs. whitespace.

    Returns
    -------
    (data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
     covariate_names, n_cov)
        See module docstring for shapes. Variable-length series are
        zero-padded to the per-dataset max length; `lengths` records the
        true per-subject count.
    """
    header, rows = _read_table(path, sep=sep)
    col_idx = {name: i for i, name in enumerate(header)}

    for required in (id_col, time_col, dv_col, dose_col):
        if required not in col_idx:
            raise ValueError(f"Column '{required}' not found in header {header}")

    if covariate_cols is None:
        reserved = {id_col, time_col, dv_col, dose_col}
        covariate_cols = [c for c in header if c not in reserved]
    n_cov = len(covariate_cols)

    # Group rows by subject id, preserving first-appearance order.
    subjects = {}
    order = []
    for row in rows:
        sid = row[col_idx[id_col]]
        if sid not in subjects:
            subjects[sid] = []
            order.append(sid)
        subjects[sid].append(row)

    nbatch = len(order)
    lengths_list = [len(subjects[sid]) for sid in order]
    T = max(lengths_list)

    data = torch.zeros(nbatch, T, 3 + n_cov, dtype=torch.float32)
    lengths = torch.zeros(nbatch, dtype=torch.int64)

    for i, sid in enumerate(order):
        subj_rows = subjects[sid]
        # sort by time, in case the table isn't already time-ordered
        subj_rows = sorted(subj_rows, key=lambda r: float(r[col_idx[time_col]]))
        n_t = len(subj_rows)
        lengths[i] = n_t
        # baseline covariates: take from the first row of the subject
        cov_vals = [float(subj_rows[0][col_idx[c]]) for c in covariate_cols]
        dose_val = float(subj_rows[0][col_idx[dose_col]])
        for t, row in enumerate(subj_rows):
            data[i, t, 0] = float(row[col_idx[time_col]])
            data[i, t, 1] = float(row[col_idx[dv_col]])
            data[i, t, 2] = dose_val
            for j, v in enumerate(cov_vals):
                data[i, t, 3 + j] = v

    lengths = lengths.int()

    dose = data[:, 0, 2].clone()
    weight_pop = data[:, 0, 3].mean() if n_cov > 0 else torch.tensor(float('nan'))
    covariates = data[:, 0, 3:].clone()

    #########################################################
    # Standardize input data (mirrors functions_theo.load_data)
    #########################################################
    data_in = data[:, :, :2].clone()
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    data_mean = data_in[:, :, 1][mask].mean()
    data_std = data_in[:, :, 1][mask].std()
    time_max = data_in[:, :, 0][mask].max()
    data_in[:, :, 0] = data_in[:, :, 0] / time_max
    data_in[:, :, 1] = (data_in[:, :, 1] - data_mean) / data_std
    # zero out padded entries so the LSTM sees a clean pad (lengths masks them anyway)
    pad_mask = ~mask
    data_in[:, :, 0][pad_mask] = 0.0
    data_in[:, :, 1][pad_mask] = 0.0

    covariates_in = covariates.clone()
    for j in range(n_cov):
        col = covariates_in[:, j]
        std = col.std()
        if std > 0:
            covariates_in[:, j] = (col - col.mean()) / std
        # if std == 0 (constant covariate), leave as-is (centered at the constant)

    return (data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
            list(covariate_cols), n_cov)


#########################################################
# Loader 2: two-file wide format (conc CSV + covariates CSV)
#########################################################
def load_two_file_wide(conc_path, cov_path, id_col='ID', dose=None, dose_col=None,
                        dose_fn=None, sample_times=None, conc_has_header='auto',
                        cov_has_header=True):
    """
    Load wide-format subject-row data: conc_path has one row per subject with
    successive concentration-measurement columns (in column order); cov_path
    has one row per subject with covariate columns (in column order).

    Parameters
    ----------
    conc_path : str
        CSV. First column is subject ID. All other columns, taken in column
        order, are successive concentration measurements (works whether
        headers are present/absent, and regardless of literal column names).
    cov_path : str
        CSV with a header row. First column is subject ID (name given by
        id_col); all remaining columns (in header order) are covariates.
    id_col : str
        Name of the subject-ID column in cov_path's header (default 'ID').
        conc_path's ID column is assumed to be its first column regardless
        of header presence/name.
    dose : float or None
        Scalar dose applied to every subject (e.g. for a fixed-dose protocol).
    dose_col : str or None
        Column name in the covariates table to use as a (per-subject) dose.
    dose_fn : callable or None
        `dose_fn(covariates_dict)` -> np.ndarray[nbatch], where
        `covariates_dict` maps covariate name -> np.ndarray[nbatch] of raw
        (unstandardized) covariate values. Use this for derived doses, e.g.
        Paclitaxel's dose = 175 mg/m^2 * BSA.
        Exactly one of {dose, dose_col, dose_fn} must be given.
    sample_times : list[float] or None
        Observation times corresponding to the concentration columns, in
        order. If None, defaults to range(n_time_cols) as floats (matches the
        conditioning_limits_PK convention `SAMPLE_TIMES = [float(t) for t in
        range(DATA_DIM)]`).
    conc_has_header : 'auto' | True | False
        Whether conc_path has a header row. 'auto' sniffs by checking if the
        first row's 2nd+ entries parse as floats.
    cov_has_header : bool
        Whether cov_path has a header row (should always be True per spec,
        but exposed for flexibility).

    Returns
    -------
    (data, data_in, lengths, dose_tensor, weight_pop, covariates, covariates_in,
     covariate_names, n_cov)
        Same contract as load_nonmem_like. Since this format provides a dense
        regular sampling grid for every subject, `lengths` is uniformly
        n_time_cols (no padding needed), matching functions_theo.load_data's
        behaviour for the fixed-grid theophylline_data.pt case.
    """
    n_dose_specs = sum(x is not None for x in (dose, dose_col, dose_fn))
    if n_dose_specs != 1:
        raise ValueError("Exactly one of {dose, dose_col, dose_fn} must be given, "
                          f"got {n_dose_specs} specified.")

    #########################################################
    # Concentration file
    #########################################################
    with open(conc_path, 'r') as f:
        first_line = f.readline().strip()
    first_fields = first_line.split(',')

    if conc_has_header == 'auto':
        has_header = False
        for v in first_fields[1:]:
            try:
                float(v)
            except ValueError:
                has_header = True
                break
    else:
        has_header = bool(conc_has_header)

    skip = 1 if has_header else 0
    conc_raw = np.loadtxt(conc_path, delimiter=',', skiprows=skip)
    if conc_raw.ndim == 1:
        conc_raw = conc_raw[None, :]

    conc_ids = conc_raw[:, 0]
    X = conc_raw[:, 1:]  # [nbatch, n_time]
    n_time = X.shape[1]

    if sample_times is None:
        sample_times = [float(t) for t in range(n_time)]
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if len(sample_times) != n_time:
        raise ValueError(f"sample_times has length {len(sample_times)}, "
                          f"but conc data has {n_time} time columns")

    #########################################################
    # Covariate file
    #########################################################
    cov_header, cov_rows = _read_table(cov_path, sep=',')
    cov_col_idx = {name: i for i, name in enumerate(cov_header)}
    if id_col not in cov_col_idx:
        raise ValueError(f"id_col '{id_col}' not found in covariates header {cov_header}")

    covariate_names = [c for c in cov_header if c != id_col]
    n_cov = len(covariate_names)

    cov_ids = np.array([float(r[cov_col_idx[id_col]]) for r in cov_rows])
    cov_matrix = np.zeros((len(cov_rows), n_cov), dtype=np.float64)
    for j, name in enumerate(covariate_names):
        col = cov_col_idx[name]
        cov_matrix[:, j] = [float(r[col]) for r in cov_rows]

    #########################################################
    # Align by subject ID
    #########################################################
    if not np.array_equal(conc_ids, cov_ids):
        # try to align by sorting/matching ID sets rather than assuming
        # identical row order
        conc_order = np.argsort(conc_ids)
        cov_order = np.argsort(cov_ids)
        if not np.array_equal(conc_ids[conc_order], cov_ids[cov_order]):
            raise ValueError("Subject IDs in conc_path and cov_path are not the same set; "
                              "cannot align.")
        X = X[conc_order]
        cov_matrix = cov_matrix[cov_order]
        conc_ids = conc_ids[conc_order]

    nbatch = X.shape[0]

    #########################################################
    # Dose
    #########################################################
    if dose is not None:
        dose_arr = np.full(nbatch, float(dose), dtype=np.float64)
    elif dose_col is not None:
        if dose_col not in covariate_names:
            raise ValueError(f"dose_col '{dose_col}' not found in covariate columns "
                              f"{covariate_names}")
        dose_arr = cov_matrix[:, covariate_names.index(dose_col)].copy()
    else:  # dose_fn
        cov_dict = {name: cov_matrix[:, j] for j, name in enumerate(covariate_names)}
        dose_arr = np.asarray(dose_fn(cov_dict), dtype=np.float64)
        if dose_arr.shape[0] != nbatch:
            raise ValueError(f"dose_fn returned shape {dose_arr.shape}, expected ({nbatch},)")

    #########################################################
    # Build `data` tensor: [nbatch, T, 3 + n_cov] = [time, conc, dose, cov...]
    #########################################################
    T = n_time
    data = torch.zeros(nbatch, T, 3 + n_cov, dtype=torch.float32)
    data[:, :, 0] = torch.from_numpy(np.broadcast_to(sample_times, (nbatch, T)).copy()).float()
    data[:, :, 1] = torch.from_numpy(X).float()
    data[:, :, 2] = torch.from_numpy(dose_arr).float().unsqueeze(1).expand(nbatch, T)
    if n_cov > 0:
        cov_t = torch.from_numpy(cov_matrix).float()
        data[:, :, 3:] = cov_t.unsqueeze(1).expand(nbatch, T, n_cov)

    lengths = torch.full((nbatch,), T, dtype=torch.int32)

    dose_tensor = data[:, 0, 2].clone()
    weight_pop = data[:, 0, 3].mean() if n_cov > 0 else torch.tensor(float('nan'))
    covariates = data[:, 0, 3:].clone()

    #########################################################
    # Standardize input data (mirrors functions_theo.load_data)
    #########################################################
    data_in = data[:, :, :2].clone()
    data_mean = data_in[:, :, 1].mean()
    data_std = data_in[:, :, 1].std()
    time_max = data_in[:, :, 0].max()
    data_in[:, :, 0] = data_in[:, :, 0] / time_max if time_max > 0 else data_in[:, :, 0]
    data_in[:, :, 1] = (data_in[:, :, 1] - data_mean) / data_std

    covariates_in = covariates.clone()
    for j in range(n_cov):
        col = covariates_in[:, j]
        std = col.std()
        if std > 0:
            covariates_in[:, j] = (col - col.mean()) / std

    return (data, data_in, lengths, dose_tensor, weight_pop, covariates, covariates_in,
            covariate_names, n_cov)
