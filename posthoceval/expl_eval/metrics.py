"""
metrics.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
import numpy as np
from scipy.special import comb

from tqdm.auto import tqdm

from posthoceval.rand import select_n_combinations
from posthoceval.rand import as_random_state

# Credit: http://www.asciify.net/ascii/show/9286
FaItHfULnEsS = '''                ixzii:`
               `MW@;`:*,
               *x@#:*.`*.
              .*`.``.+`.*
              ,;`   `,*`i.
              ,; ` ```*,.*
              .i```````+`*`
               *```````:i,;
               i.`;*````+`+`
               `*`;n``` :i,;
                ;,.+.;.``+.*`
                 *.*n,```,+`*
                 :;`*.``,`i;::                          `.`
                  *.,*`:i``*,*`       .:.     .;i;,`  .i+*+.
            `.`   .+**:*,` `+:#:.` `,**i**i:;*+i;;i++*+i;ii+.
           ,*i***iiz,*z;````i+z;i***i;;;;;iii*+z#+;;;;;:;*;ii
          `*;;i*****#```````;@M:;;+#z+;;;;;;;znxnx*;;;ii*i;;*
          :i;+nnnxnnxi``````n@@+:*nnnn;;;;;;;*##zzi;;iii;;;;*
      `.:i*;;*nnnxxxnz*.``:n@@xx:;i**++++i;;;;;;;;;;;*;;;;;ii
     `**ii;;;;;ii*+*i;ixxW@@@x*x;;i#nnnxnnn+;;;;;;;;;*;;;;;i:
     .*;;;;;;;i+#+*;;;;*@@@Wni*x;;#xnnnnnnnni;;;;;;;ii;;;;;*.
  `,;i;;;;;;;;+xxxx*;;;#z#+*iinn;;i#nnnnnnz+;;;;;;**i;;;;;;i:
 ,+*i;;;;;;;;;;*+##i;+n#iiiiinn+;;;;;iiii;;;;;;;;i;;;;;;;;;;*.
:+*+*;;;;*+++*;;;;i#nz*iiii;z#ni*i;;;i*ii*i;;;i**+;;;;;;+z+;;*.
*;;;i++++i;:;i+++*n#*iiiii;z#z+;;i***i;;;;;iiii;;;i;;;;znnn+:;*
+::::;;;;::::::;::n*iiiii*n##z:::::;::::i++*;::::;*;;;*xnnnn;:+`
;i:;++++*;::::::::##++*+zn+zz:::::::::;i*###;:::::i;;;zxnnnn;;*
 :i;+##+*;::::::::;zzzzz##n*::::::::;++*i;;i++;:::i;;innnnnn;;*                 ,,
  .*;;;;*;:::::::::;+znnz*;::::::::;+i:i:::;:;*i::i;;innxnxz;;*               `#+z.
   .++;;z:::*::::::::::::::::::::::;;::n::;z:::;::i;;;nxnnx+;;*              ,#;:+,
   ,#:::n;:;n:::::::::::::::::::::*i:::z*i*n::::::*;;;+nxxz;;;*`            :#;::#.
  ,#z;;+x###x*::::::::::::::::::::izi##+ii*+#+::::*;;;;*#+;;;;i;           :#;:i:#`
  #;in+,````.i#;:::::::::::::::::::n+.```` ``:#;::ii;;;;;;;;;;;*i    `.`  :+::n*:+
 `#:#;````````,z;:::::::::::::::::#:``   ``  `,#::;*i;;;;;;i*;;;*:  :*ii++z::++;:+
  ##;.;i,` `  `:#::::::::::::::::+;``    `.;:``*i::;ii;;;;;zx+;;ii :;`  `.#i+.+::+   ``
  ,#,z+*#+`    `#;:::::::::::::::#```  ` iz+#z.,#:::;*;;;;ixx#;;ii.*```   .n, #::* +#zz`
  i,#+xz;#:    `*i::::::::::::::;+      .z*x#*+`z:::;;i;;;*xx*;;i*+```    `*.`#:;i,+#;#,
  *.zn#@i+*    `ii::::::::::::::i*``    ;#z#@iz.#::::;*;;;in+;;;;z,``     `;:`#:i;;iz;#z,
  *.##Mniz,    `#;::::::::::::::;*```   ,z*Wn*+i*::::;*;;;;;;;;;;#``      `i:.+:*+*;##i;i
  ;:.z#+zi``  `,#::::::::::::::::z``     *z++z,z:::::;*;;;;;;;;;#:``      ,+`:*:#z#+*z:i;
  `+``:;.`` ` .+;:;::::::::::::::**``    `,i;,zi::::::*;;;;;;;*#n``      ,*` ;i:z*iz:z:+,
   ,*` `    `.+i:;+:ii::::::::::::++.`  ```,+z;:::::::*;;;;;;inxz       ;i`  i;:nii;*+:#`
    ,+,````.i#i:;#i:#;:::::::::::::i##+++###i::*::::::i;;;;;;zxx+     `#xnnnnM;:z:;;+:ii
     .#z##zz*::;zi::z:::::::::::::::ii;;;;:::i+i;::;i:;*i;;;innx#   `;nz#+###x;;+*#:::#`
      `#z+i;::izi::i#:::::::::::::::;*++***++*;:ii:;i::;ii;;+nnnx:,iz++#####+x;:iz;::+:
        ;+::;##::::#;:::::::::::::::::;;;;;;::::;;;i::::;i;;+xnnxn#*i:;nxx#xMMi:+*::**
        `*###i::::;#::::::::::::::::::::::::::::::;;:::::;i;*xnnn*;;;;;ii+*+:nz:;;:*M#
       `+z*;:::::;#;:::::::::::::::::::::::::::::::::::::;i;;nxx#;;;;;;;;+i..nMz*+nMxM`
       i*::::::::+*:i*::::::::::::;::::::::::::::::::::::;i;;i+*;;;;*#;;;+,..#M@@Wxxxxi
       +;::::::;*+::i+::::::::::::::::;::::::::::::::::::;*;;;;;;;;;nx*;;#...iMW@@xxxxn
       +::::::;++:::*+::::::::::;*znxxxxn+;:::::::::::::::ii;;;;;;;ixx*;;+...,MM#@xxxxM.
       :#;::;*#*::::;i::::::::inWWWMMMMMWW#:::::::::::::::;i**;;;;;;nn;;;+;...nM##xxxxx*
        ,+##n#;:::::::::::::;nWWMWMWMMWWWM+:::::::::::;:::::;;*;;;;;;;;;;;+...+M@@Mxxxx#
            *;:::::::::::::+MWMMMMWWMn#*i;:::::::;i*++++*;:::;;i;;;;;;;;;;#...;MW#WxxMxi
            *;:::::::::::;nWWMMMWMz*;::::::;::::;++++###++;:::;*;;;;;;;;;i*...,xMWxxn;
            *;::::::::::;xWWWWWn*;::::::;*#z;:::i#+++#++#+;::::*;;;;;;;;;i*....zxxM*`
            *;:::::::::;z*MWWzi:::::;*#zz+i::::::*++++++*;:::::ii;;;;;;;;i+....iMn,
            *;::::::::;n*:nz;::::i#z#*;;:::::::::::;;:::::;ii;:;i**ii;;;;;#,...iz`
            ii::::::::+izn*:::i#z+i::::::::::::::::::::::i+##+;:;:::*;;;i;*;.;+i.
            ;+:::::::*:z;;::*nzi:::::::::::::::::::::::::i##++;:::::;**ii*i##i`
            `z:::::::*`z;;*#;.i*i;:::::::::::::::::::::::;iii;:::::::;;;;;i*`
             *i:::::*, .+#i`   `,i*i;:::::::::::::::::::::::::::::::;;i+zi.
             `#:::::+             `zxz+*;;:::::::::::::::::::::;i*+#zxxxM`
              ;*:::i:            `zMxxxxxnz#+**iiii;;iii***iiixxxxxxxxxxn
               +;::+             #Mxxxxxxxxxn``..,,,,,,.``   +xxxxxxxxxx+
               `+;:i            `MMMMMxxxxxMi                xxMMMMxxxxM:
                .#+.             iMM#izMxxxM`                xMM#:zMMxxx`
                 `*              `z*:inMxxMi                 ,#M;:xxMxMi
                                `+i:*+znxx*                   `#:in##+;
                               `+;:+i                         ,*:+:
                               ;i,in.                         *;:z`
                               .#i+*#+.                       #:;#
                               `iMx+i;+i`                    .+:in:`   ,*,
                              `n@@@@M;.:#:                   `#:#i+z+::@#@;
                              i####@@@n*#@+                   ,#+*;:,izW##@;
                              x##WW#######n                     :+z:.`.i###@.
                              x##@@#######n                       `i#*,+####*
                              +###########+                         `+W@####:
                              `M#####xWWx*`                      `,,.i#####z
                               .z@@W+`                          ,M@#@@###@x`
                                 ``                             x#@W#####n`
                                                                W#@@#####i
                                                                n######@#`
                                                                `+xWWn+.'''


def sensitivity_n(model, attribs, X, n_subsets=100, max_feats=0.8,
                  n_samples=None, n_points=None, aggregate=True,
                  seed=None, verbose=False):
    """
    Pros:
    - can quantify the number of features attribution methods are capable of
      identifying "correctly"
    - objectively gives measure of goodness in terms of target variation, it
      at least weeds out completely off attributions
    Cons:
    - only works on feature attributions, so no interactions or other types of
      explanations
    - assumes a removed value is zero, not truly removed, 0 may not be best
      choice
    - assumes variation is linear with attribution, but attributions assume
      local linearity, locality can be breached by setting features to 0

    Aggregated version of sensitivity-n
    Max score of 1, minimum score of -1 (validate this min...)

    "Towards better understanding of gradient-based attribution methods for
     Deep Neural Networks"

    :param model:
    :param X:
    :param n_subsets:
    :return:
    """
    assert X.ndim == 2, 'i lazy gimme 2 dims for X'
    assert len(X) == len(attribs)

    n_feats = X.shape[1]
    feats = np.arange(n_feats)

    if n_points is None:
        n_points = min(16, n_feats)
    if n_samples is None:
        n_samples = min(1000, len(X))

    rs = as_random_state(seed)

    sample_idxs = rs.choice(np.arange(len(X)), n_samples, replace=False)
    X_eval = X[sample_idxs]
    attribs_eval = attribs[sample_idxs]

    # try to include at least two values of n
    max_n = max(n_feats * max_feats, min(2, n_feats))
    all_n = np.unique(np.round(
        np.linspace(1, max_n, n_points)).astype(int))

    # pearson corr coefs
    all_pccs = []

    # gather all explanations
    if verbose:
        tqdm.write('Calling model with evaluation data')
    y_eval = model(X_eval)

    pbar_n = tqdm(enumerate(all_n), total=len(all_n),
                  desc='N', disable=not verbose, position=0)
    pbar_x = tqdm(total=len(X_eval), desc='X', disable=not verbose,
                  position=1)

    bad_n = []

    for n_idx, n in pbar_n:
        pbar_n.set_description(f'N={n}')

        pccs = []

        pbar_x.reset()
        for x_i, y_i, attrib_i in zip(X_eval, y_eval, attribs_eval):
            pbar_x.update()

            # TODO: descriptions only here for debug - update less....
            pbar_x.set_description('Select combinations')

            max_combs = comb(n_feats, n, exact=True)
            n_subsets_n = min(max_combs, n_subsets)
            # corr not defined for <2 points...
            if n_subsets_n < 2:
                continue
            combs = select_n_combinations(feats, n, n_subsets_n, seed=rs)

            # model output for sample
            pbar_x.set_description('Call model')

            # Create array of all perturbations of x
            pbar_x.set_description('Permute x')

            all_x_s0s = np.repeat(x_i[None, :], len(combs), axis=0)
            idx_rows = np.arange(len(combs))[:, None]
            idx_feats = np.asarray(combs)
            all_x_s0s[idx_rows, idx_feats] = 0

            # explain samples and compute attribution sum
            pbar_x.set_description('Explain')

            attrib_sum_subset = attrib_i[idx_feats.ravel()].reshape(-1, n)
            attrib_sum_subset = attrib_sum_subset.sum(axis=1)

            # compute model output for perturbed samples
            pbar_x.set_description('Call model (permuted)')

            all_y_s0s = model(all_x_s0s)
            # Handle the cases where output may be multi-class
            if all_y_s0s.ndim == 2:
                n_class = all_y_s0s.shape[1]
                if n_class == 1:
                    all_y_s0s = all_y_s0s.squeeze(axis=1)
                else:
                    y_shape = np.shape(y_i)
                    predicted_idx = np.argmax(y_i)

                    y_i = y_i[predicted_idx]
                    all_y_s0s = all_y_s0s[:, predicted_idx]

            invalid_mask = np.isnan(all_y_s0s) | np.isinf(all_y_s0s)

            if invalid_mask.any():
                all_y_s0s = all_y_s0s[~invalid_mask]
                if len(all_y_s0s) < 2:
                    continue
                attrib_sum_subset = attrib_sum_subset[~invalid_mask]

            all_y_diffs = y_i - all_y_s0s

            # compute PCC
            pccs.append(
                np.corrcoef(all_y_diffs, attrib_sum_subset)[0, 1]
            )
        pbar_x.refresh()

        if not len(pccs):
            bad_n.append(n_idx)
            continue

        # append average over all PCCs for this value of n
        mean_pcc = np.mean(pccs)
        all_pccs.append(mean_pcc)
    pbar_x.close()

    if len(bad_n):
        all_n = np.delete(all_n, bad_n)

    if aggregate:  # AUC normalized to [-1,1]
        if not len(all_pccs):
            return np.nan
        if len(all_pccs) == 1:
            return all_pccs[0]
        # all_n[-1] is max, 1 is min
        return np.trapz(x=all_n, y=all_pccs) / (all_n[-1] - 1)
    else:
        return all_n, all_pccs


def faithfulness_melis(model, attribs, X, ref_val=0, aggregate=None,
                       drops_only=None, verbose=False):
    """
    TODO: ref_val=mean version of this, values computed from X

    FaItHfULnEsS

    Naming of this metric is adapted from paper and questionable
    TODO- maybe rename this...

    No NaN removal is performed unless aggregating to a scalar

    References:
        .. [#] `David Alvarez Melis and Tommi Jaakkola. Towards robust
           interpretability with self-explaining neural networks. In S. Bengio,
           H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R.
           Garnett, editors, Advances in Neural Information Processing Systems
           31, pages 7775-7784. 2018.
           <https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf>`_

       https://github.com/dmelis/robust_interpret/blob/master/robust_interpret/explainers.py#L292

    Args:
        model: callable model
        X (numpy.ndarray): data.
        attribs (numpy.ndarray): attributions of features per sample of X
        ref_val ((numpy.ndarray): array or scalar (default=0)
    Returns:
        float: correlation between attribute importance weights and
            corresponding effect on classifier.
    """
    # TODO: note that this is pretty much identical to sensitivity-n with minor
    #  changes, e.g., not taking mean of each feature corr, otherwise
    #  sensitivity-n can really be thought of as a superseding function
    if verbose:
        print('"FaItHfULnEsS"')
        print(FaItHfULnEsS)

    if aggregate is None:  # nothing specified
        if drops_only:  # no aggregation at all, return prob drops
            aggregate = False
        else:
            # The default setting - take means of PCCs
            aggregate = True
            drops_only = False
    elif aggregate:  # we will aggregate PCCs using mean
        if drops_only:
            raise ValueError('drops_only cannot be True if aggregate is True')
        else:
            drops_only = False
    else:  # we will not fully aggregate
        if drops_only is None:  # by default just return PCCs
            drops_only = False
        # otherwise we take mean or return prob drops according to drops_only

    assert X.ndim == 2, 'i lazy gimme 2 dims for X'
    assert len(X) == len(attribs)

    n_feats = X.shape[1]
    feats = np.arange(n_feats)

    # make to array
    ref_val_shape = np.shape(ref_val)
    if not ref_val_shape:
        ref_val = np.repeat(ref_val, n_feats)
    else:
        assert len(ref_val_shape) == 1 and ref_val_shape[0] == n_feats

    # obtain model outputs
    y = model(X)

    # Handle the cases where output may be multi-class
    predicted_idxs = None
    if y.ndim == 2:
        n_class = y.shape[1]
        if n_class == 1:
            y = y.squeeze(axis=1)
        else:
            predicted_idxs = np.argmax(y, axis=1)
            y = y[predicted_idxs]

    # probability drops per feature
    prob_drops = []

    for feat_idx in feats:

        X_for_i = X.copy()
        X_for_i[:, feat_idx] = ref_val[feat_idx]

        y_for_i = model(X_for_i)

        if predicted_idxs is not None:
            y_for_i = y_for_i[:, predicted_idxs]

        prob_drops.append(y - y_for_i)

    # combine into N x d matrix of prob drops
    prob_drops = np.stack(prob_drops, axis=1)

    if drops_only:
        return prob_drops

    # compute PCCs
    pccs = []
    invalid_mask = np.isnan(prob_drops) | np.isinf(prob_drops)
    for row, mask, attribs_row in zip(prob_drops, invalid_mask, attribs):
        mask = ~mask
        row = row[mask]
        if len(row) < 2:
            if not aggregate:
                pccs.append(np.nan)
        else:
            pccs.append(
                np.corrcoef(row, attribs_row[mask])[0, 1]
            )

    if aggregate:  # TODO: 2 type of non-agg and 1 type of overall agg
        if not len(pccs):
            return np.nan
        return np.mean(pccs)
    else:
        return pccs
