import numpy as np
import os
import matplotlib.pyplot as plt
from pycse import regress

mcolors = {'Rh': '#8073B1',
           'Ir': '#C84B54',
           'Pd': '#138198',
           'Pt': '#27ae60',
           'Ag': '#95a5a6',
           'Au': '#f1c40f'}

amarkers = {'C': 'o',
            'N': '^',
            'O': 's',
            'F': 'd',
            'CH': 'v',
            'CH2': '8',
            'HN': 'h',
            'H2N': '<',
            'HO': '>'}


def add2db(db, atoms, **kwargs):
    '''
    Adds entry to database object if no existing db entry with the
    given kwargs is present.
    This way we avoid duplication in the database.
    Requires existing connection to a db file
    '''
    try:
        db.get(**kwargs)
    except(KeyError):
        db.write(atoms, **kwargs)


def get_band_properties(energies, dos, energy_range=(-10, 5)):

    '''
    Return the band center, band width, total # of states,
    # of occupied states, and fractional filling.

    Energies and dos must be a list or numpy array. If energy_range
    is provided, then computations only include energies and dos
    within that range.
    '''

    if energy_range is not None:
        ind = (energies < energy_range[1]) & (energies > energy_range[0])
        energies = energies[ind]
        dos = dos[ind]

    Nstates = np.trapz(dos, energies)
    occupied = energies <= 0.0
    N_occupied_states = np.trapz(dos[occupied], energies[occupied])
    e_band = np.trapz(energies * dos, energies) / Nstates
    w_band2 = np.trapz(energies**2 * dos, energies) / Nstates
    w_band = np.sqrt(w_band2)
    frac_filling = N_occupied_states / Nstates

    return e_band, w_band, Nstates, N_occupied_states, frac_filling


def plot_dos(energies,
             dos,
             energy_range=(-10, 5),
             c=None,
             label=None,
             xlabel=True,
             ylabel=True,
             alpha=0.3,
             keep_x_ticks=True,
             keep_y_ticks=False,
             plot_center=True,
             h=0.3,
             w=8,
             fontsize=28):

    '''
    Makes a plot the DOS on the current axis.
    The occupied states are shaded.
    alpha controls the intensity of the shading.
    Assumes DOS is referenced to E_fermi.
    If plot_center is True, it plots the center of the band
    as a rectangular point of height=h and width=w.
    fontsize controls the fontsize of the labels.
    '''

    if energy_range is not None:
        ind = (energies < energy_range[1]) & (energies > energy_range[0])
        energies = energies[ind]
        dos = dos[ind]

    occupied = energies <= 0.0

    plt.plot(energies, dos, c=c, label=label, lw=5)
    plt.fill_between(x=energies[occupied],
                     y1=dos[occupied],
                     y2=np.zeros(dos[occupied].shape),
                     color=c,
                     alpha=alpha)
    if plot_center:
        band_center = get_band_properties(energies, dos)[0]
        plt.plot([band_center, band_center], [0, h], c=c, lw=w)

    if not keep_x_ticks:
        plt.xticks([])

    if not keep_y_ticks:
        plt.yticks([])

    if xlabel:
        plt.xlabel('$E - E_\mathrm{F}$ (eV)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize-4)

    if ylabel:
        plt.ylabel('DOS', fontsize=fontsize)


def get_energy(db, **kwargs):

    try:
        energy = db.get(**kwargs).get('energy', np.nan)
    except(KeyError):
        energy = np.nan

    return energy


def get_row_energies(rows,
                     metals=['Rh', 'Ir', 'Pd', 'Pt', 'Ag', 'Au'],
                     exclude_site_change=True):

    '''Loop through an atoms row object to populate a np.array with
    energies sorted according to the order in metals'''

    # Note: metals is defined as a global
    metal_indices = dict(zip(metals, range(len(metals))))
    Es = np.ones(len(metals)) * np.nan
    for row in rows:
        m = ''.join(set(row.symbols).intersection(metals))

        Es[metal_indices[m]] = row.energy

        try:
            if exclude_site_change and row.site_change:
                Es[metal_indices[m]] = np.nan
        except AttributeError:
            pass

    return Es


def get_E_ads(db, a, a_s, c_s):
    '''
    Return an array of adsorption energies for
    different metals
    '''
    arows_relax = db.select(a_s)
    arows_clean = db.select(c_s)

    a_relax_es = get_row_energies(arows_relax)
    a_clean_es = get_row_energies(arows_clean)

    a_e = db.get(specie=a).energy

    return a_relax_es - a_clean_es - a_e


def get_pair_E_ads(db, a1, a2, a1_s, a2_s, c1_s, c2_s=None):

    '''Return adsorption energies of a1 and a2 for atoms rows
    returned by a1_select and a2_select (surface plus adsorbate),
    c1_select and c2_select are the clean surfaces corresponding to
    a1 and a2 (if different than a1).
    '''

    a1rows_relax = db.select(a1_s)
    a2rows_relax = db.select(a2_s)

    a1_relax_es = get_row_energies(a1rows_relax)
    a2_relax_es = get_row_energies(a2rows_relax)

    a1rows_clean = db.select(c1_s)
    a1_clean_es = get_row_energies(a1rows_clean)

    if c2_s is not None:
        a2rows_clean = db.select(c2_s)
        a2_clean_es = get_row_energies(a2rows_clean)
    else:
        a2_clean_es = a1_clean_es

    a1_e = db.get(specie=a1).energy
    a2_e = db.get(specie=a2).energy

    a1_E_ads = a1_relax_es - a1_clean_es - a1_e
    a2_E_ads = a2_relax_es - a2_clean_es - a2_e

    return a1_E_ads, a2_E_ads


def print_possible_kvp(db, selection, limit=None):

    keys = {}
    atomsrows = db.select(selection)
    print 'Possible key-value pairs for selection: {0}'.format(selection)
    for arow in atomsrows:
        for k, v in arow.key_value_pairs.iteritems():
            if k not in keys:
                keys[k] = set()
            keys[k].add(v)

    for k, v in keys.iteritems():
        val = ", ".join(str(e) for e in list(v))
        print '{k}: {val}'.format(**locals())
    print


def regression(E1, E2, alpha=0.05):

    '''Perform regression but remove nan values'''

    xy = np.transpose(np.array([E1, E2]))
    xy = xy[~np.any(np.isnan(xy), axis=1)]
    x = xy[:, 0]
    A = np.column_stack((x ** 1, x ** 0))
    y = xy[:, 1]

    pars, pint, se = regress(A, y, alpha)

    y = np.dot(A, pars)
    return x, y, pars, pint, se


def get_colors(colors, E1, E2):

    '''
    Remove colors with nan values
    '''
    colors = np.array(colors)
    ab = np.array([E1, E2])
    new_colors = colors[~np.any(np.isnan(ab), axis=0)]
    return list(new_colors)
