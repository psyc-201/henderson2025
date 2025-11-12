% Simulation: Making 36 new blobs!
% Utilizing some code from Henderson 2025 repo
% L. Kemmelmeier, 2025
%
% Picks a "kind" (base rfc params for blobs)
% Varies 2 rfcs (x,y) over 6x6 grid with inset coords
% Generates individual images + grid of them all
% Saves csv of filenames for images (includes x, y coords)

%% Init variables + set paths
kind = 'snm'; % 'balls','snm','fetus','cater','peanut','test'
rfc_x_idx = 2; % rfc along x axis (1..7)
rfc_y_idx = 5; % rfc along y axis (1..7)

out_root = '/Users/lenakemmelmeier/Documents/GitHub/henderson2025/stimuli';
out_name = 'new_blob_stim';

img_size = 224; % even nums only
bg_gray = 0.30; % background gray
fg_gray = 0.90; % shape gray
save_silhouette = true; % write PNGs? true is on

grid_ticks = [0.10 1.06 2.02 2.98 3.94 4.90]; % inset ticks on 0..5
amp_boost = 2.0; % multiply step size for more dramatic rfc changes
debug_plot_n = 0; % quick visual debug (0 = off)

% output a csv w/ the image coords
save_dir = fullfile(out_root, out_name);
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
manifest_csv = fullfile(save_dir, 'manifest_coords.csv');

% base params (parameter set from Op de Beeck / Drucker, used in Henderson repo)
par = odb_params(kind);
assert(mod(img_size,2) == 0, 'img_size must be even.');

% evenly map 0–5 coords to rfc amplitude range
n_dim = numel(grid_ticks);
new_step = amp_boost .* (5 .* par.step ./ (n_dim - 1)); % (5*step)/(n_dim-1) * amp_boost

[grid_x, grid_y] = meshgrid(grid_ticks, grid_ticks);
coords = [grid_x(:), grid_y(:)];
n_stim = size(coords,1);

%% Generate and save
fid = fopen(manifest_csv, 'w');
fprintf(fid, 'index,filename,x_val,y_val\n');

for ii = 1:n_stim
    x_val = coords(ii,1);
    y_val = coords(ii,2);

    % makes one silhouette w/ chosen rfc pair
    % non-integer freq handling + endpoint correction from Op de Beeck et al. (2001, 2003)
    img_bin = generate_one_blob(par, img_size, new_step, rfc_x_idx, rfc_y_idx, x_val, y_val);

    % paint grayscale
    img_gray = double(img_bin);
    img_gray(img_gray == 0) = bg_gray;
    img_gray(img_gray == 1) = fg_gray;

    fn = sprintf('blob_%.2f_%.2f.png', x_val, y_val);
    fpath = fullfile(save_dir, fn);

    if save_silhouette
        imwrite(img_gray, fpath);
    end

    fprintf(fid, '%d,%s,%.2f,%.2f\n', ii, fn, x_val, y_val);

    if debug_plot_n > 0 && ii <= debug_plot_n
        figure;
        imshow(img_gray, []);
        title(sprintf('%s  x=%.2f  y=%.2f', kind, x_val, y_val));
        drawnow;
    end
end

fclose(fid);

%% Make grid of all the blobs
tile = img_size;
gap_px = 8;
border_px = 16;
c_dim = n_dim;

H = c_dim * tile + (c_dim - 1) * gap_px + 2 * border_px;
W = c_dim * tile + (c_dim - 1) * gap_px + 2 * border_px;

canvas = ones(H, W) * bg_gray;
row_to_pix = @(r) (border_px + (r - 1) * (tile + gap_px) + (1:tile));
col_to_pix = @(c) (border_px + (c - 1) * (tile + gap_px) + (1:tile));

for r = 1:c_dim
    for c = 1:c_dim
        idx = (r - 1) * c_dim + c;
        x_val = coords(idx,1);
        y_val = coords(idx,2);

        fn = sprintf('blob_%.2f_%.2f.png', x_val, y_val);
        img_gray = im2double(imread(fullfile(save_dir, fn)));

        if size(img_gray,3) == 3
            img_gray = rgb2gray(img_gray);
        end

        rr = row_to_pix(r);
        cc = col_to_pix(c);
        canvas(rr, cc) = img_gray;
    end
end

imwrite(canvas, fullfile(save_dir, 'grid_montage.png'));

%% Helper function defs
    function img_bin = generate_one_blob(par, img_size, new_step, rfc_x_idx, rfc_y_idx, x_val, y_val)
    % rfc contour → binary silhouette for one (x,y)
    % radius modulated by sinusoidal rfc components (Op de Beeck et al., 2001, 2003)
    % fill step (roipoly/imfill) adapted from Drucker and Op de Beeck MATLAB code
    % smoothing + normalization consistent with Henderson 2025 implementation

        w = par.w;
        A = par.A;
        P = par.P;
        step = par.step;

        sz_adj = img_size / 299;
        A = A .* sz_adj;
        step = step .* sz_adj;
        new_step = new_step .* sz_adj;

        center = img_size / 2;
        r_px = round(70 * sz_adj);
        n_pix = round(2 * pi * r_px);

        amp_step = zeros(size(A));
        amp_step(rfc_x_idx) = new_step(rfc_x_idx) * x_val;
        amp_step(rfc_y_idx) = new_step(rfc_y_idx) * y_val;

        % radius as function of angle (sum of RFC sinusoids)
        curve_func = @(theta) r_px + sum((A + amp_step) .* sin(w .* theta' + P), 2);

        % endpoint closure correction for non-integer frequencies (Op de Beeck fix)
        y_begin = curve_func(0);
        y_end = curve_func(2 * pi);
        delta_close = y_end - y_begin;

        x_poly = [];
        y_poly = [];

        for kk = 0:n_pix
            theta = 2 * pi / n_pix * kk;
            rho = curve_func(theta);

            y_valc = rho * cos(theta);
            x_valc = (-1) * rho * sin(theta);
            y_valc = y_valc + cos(theta / 2) * delta_close / 2;

            x_coord = round(x_valc + center);
            y_coord = round(y_valc + center);

            if kk > 0
                x_range = [x_poly(end), x_coord];
                y_range = [y_poly(end), y_coord];
                n_ins = max(abs([diff(x_range), diff(y_range)])) + 1;

                x_i = linspace(x_range(1), x_range(2), n_ins)';
                x_i = round(x_i(2:end));

                y_i = linspace(y_range(1), y_range(2), n_ins)';
                y_i = round(y_i(2:end));

                x_poly = [x_poly; x_i];
                y_poly = [y_poly; y_i];

            else
                x_poly = [x_poly; x_coord];
                y_poly = [y_poly; y_coord];
            end
        end

        % light smoothing to reduce jagged edges (seen in Henderson code)
        win_sz = max(1, floor(5.84 * sz_adj));

        if win_sz > 1
            win = ones(1, win_sz);
            x_poly = filtfilt(win / win_sz, 1, x_poly);
            y_poly = filtfilt(win / win_sz, 1, y_poly);
        end

        img_bin = roipoly(zeros(img_size), x_poly, y_poly);
        img_bin = imfill(img_bin, 'holes');
        img_bin = img_bin';
    end

    function par = odb_params(kind)
    % rfc base parameter sets from Op de Beeck and Drucker (used in Henderson repo)
        if nargin == 0
            kind = 'balls';
        end

        params.balls.w = [0.55 1.11 4.94 3.39 1.54 3.18 0.57];
        params.balls.A = [20.34 2.00 18.83 24.90 2.00 15.38 21.15];
        params.balls.P = [3.20 3.64 2.79 3.86 3.51 0.34 1.08];
        params.balls.step = [0 6.00 0 0 6.00 0 0];

        params.fetus.w = [1.79 2.65 2.82 4.66 1.67 1.13 2.65];
        params.fetus.A = [5.97 2.00 1.30 10.60 2.00 11.60 16.00];
        params.fetus.P = [1.88 1.79 6.21 3.24 2.72 4.78 1.31];
        params.fetus.step = [0 5.17 0 0 6.00 0 0];

        params.snm.w = [1.92 1.73 4.82 7.33 2.57 2.11 2.77];
        params.snm.A = [22.50 2.00 1.68 15.00 2.00 19.20 7.30];
        params.snm.P = [4.65 3.99 5.94 1.59 4.60 0.45 5.39];
        params.snm.step = [0 6.00 0 0 3.44 0 0];

        params.cater.w = [1.29 3.48 0.82 5.34 2.11 1.96 1.83];
        params.cater.A = [29.00 2.00 16.40 18.40 2.00 16.30 11.30];
        params.cater.P = [4.18 0.06 2.70 4.32 5.38 2.90 2.59];
        params.cater.step = [0 5.18 0 0 5.18 0 0];

        params.peanut.w = [0.00 1.70 0.00 0.00 2.11 0.00 0.00];
        params.peanut.A = [0.00 10.00 0.00 0.00 10.00 0.00 0.00];
        params.peanut.P = [0.00 3.00 0.00 0.00 5.38 0.00 0.00];
        params.peanut.step = [0 6.00 0 0 4.97 0 0];

        params.test.w = [0.00 1.70 0.00 0.00 2.11 0.00 5.00];
        params.test.A = [0.00 5.00 0.00 0.00 10.00 0.00 11.00];
        params.test.P = [0.00 3.00 0.00 0.00 5.38 0.00 3.00];
        params.test.step = [0 7.00 0 0 8.97 0 0];

        par = params.(kind);
    end