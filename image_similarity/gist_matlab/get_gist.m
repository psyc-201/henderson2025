function [] = get_gist(debug, overwrite)

    if nargin==0
        debug=0
        overwrite=0
    end
    
    debug = logical(debug);
    overwrite = logical(overwrite);
    
    fprintf('debug=%d, overwrite=%d\n', debug, overwrite)
    
    mypath = pwd;
    filesepinds = find(mypath==filesep);
    nDirsUp = 1;
    root = mypath(1:filesepinds(end-nDirsUp+1));

    
    feat_dir = fullfile(root, 'features');
    images_filename = fullfile(feat_dir, 'images_all.h5py');
    save_dir = fullfile(feat_dir, 'gist');
    
    fprintf('Reading image brick from %s...\n', images_filename)
    I = h5read(images_filename, '/stimuli');
    [h, w, c, n_total_ims] = size(I);
    fprintf('size of image brick:\n')
    disp([h,w,c,n_total_ims])

    if debug
        I_batch = I(:,:,:,1:10);
    else
        I_batch = I;
    end
    
    fprintf('size of batch:\n')
    disp(size(I_batch))

 
    param.imageSize = 224;
    %param.orientationsPerScale = [8 8 8 8];
    param.orientationsPerScale = [4 4 4 4];
    param.numberBlocks = 4;
    %param.numberBlocks = 2;
    param.fc_prefilt = 4;
    
    [gist, param] = LMgist(I_batch, 0, param);

    fprintf('size of gist output:\n')
    disp(size(gist))
    disp(param)
    
    if ~exist(save_dir, 'dir')
       mkdir(save_dir)
    end
    
    save_gist_filename = fullfile(save_dir, 'images_gistdescriptors_4ori_4blocks.mat')
    fprintf('saving to %s\n', save_gist_filename);
    save(save_gist_filename, 'gist','param');
    
    % put into the format we will need later, which will be [n_images x n_features]
    % the dims get reversed from matlab to python, so needs to be opposite order
    [n_images, n_features] = size(gist);
    gist_reshaped = reshape(gist', [n_features, n_images]);
    
    save_gist_filename_h5 = fullfile(save_dir, 'images_gistdescriptors_4ori_4blocks.h5py')
    if exist(save_gist_filename_h5,'file')
        if overwrite
            fprintf('File exists, removing it now...\n')
            unix(char(sprintf('rm %s',save_gist_filename_h5)));
        else
            error('File already exists. To overwrite it set overwrite=True.');
        end
    end
    fprintf('saving to %s\n', save_gist_filename_h5);
    h5create(save_gist_filename_h5,'/features',[n_features, n_images]);
    h5write(save_gist_filename_h5, '/features', gist_reshaped, [1,1], size(gist_reshaped));


end




