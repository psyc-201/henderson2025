import numpy as np

def get_main_grid():
    
    start = 0;   
    stop = 5;
    nsteps_main = 4;    
    start_grid = 0.1;
    stop_grid = 4.9;
    main_pts = np.round(np.linspace(start_grid,stop_grid, nsteps_main),1)
    [gridx,gridy] = np.meshgrid(main_pts,main_pts);
    main_grid_points = np.array([gridx.ravel(),gridy.ravel()]).T;

    return main_grid_points

def get_quadrant(grid_pts, center = 2.5):
    
    # quadrants are numbered counter-clockwise from top right
    
    npts = grid_pts.shape[0]
    quad = np.zeros((npts,))-1
    
    quad[(grid_pts[:,0]>center) & (grid_pts[:,1]>center)] = 1
    quad[(grid_pts[:,0]<center) & (grid_pts[:,1]>center)] = 2
    quad[(grid_pts[:,0]<center) & (grid_pts[:,1]<center)] = 3
    quad[(grid_pts[:,0]>center) & (grid_pts[:,1]<center)] = 4

    return quad.astype(int)

def get_categ(grid_pts, task_num):

    assert(np.isin(task_num, [1,2,3]))
    # tasks are:
    #  ['Decode: Linear (1)','Decode: Linear (2)','Decode: Checker'];
    ti = task_num-1
    # NOTE the category labels (1&2) are swapped here relative to in the experiment script
    # in these, category 1 = coord<center and category 2 = coord>center
    # this ordering seems more logical to me
    quad_groups = [[[2, 3], [1, 4]],
                    [[3, 4], [1, 2]],
                    [[2, 4], [1, 3]]];
    # quad_groups = [[[1, 4], [2, 3]],
    #                 [[1, 2], [3, 4]],
    #                 [[1, 3], [2, 4]]];

    quad = get_quadrant(grid_pts)
    categ_labs_task = np.zeros_like(quad)
    categ_labs_task[np.isin(quad, quad_groups[ti][0])] = 1
    categ_labs_task[np.isin(quad, quad_groups[ti][1])] = 2
    
    return categ_labs_task

def get_dist_from_bound(grid_pts, task_num, center = 2.5):

    assert(np.isin(task_num, [1,2,3]))
    # tasks are:
    #  [Linear (1)','Linear (2)','Checker'];
    
    if task_num<3:
        ti = task_num-1
        dist = np.abs(grid_pts[:,ti]-center)
    else:
        dist = np.min(np.abs(grid_pts-center), axis=1)
        
    return dist

def get_dist_from_center(grid_pts, center = 2.5):

    dist = np.sqrt(np.sum((grid_pts-center)**2, axis=1))
    
    return dist


def get_full_grid():
    
    start = 0;   
    stop = 5;
    step = 0.1;
    center = (stop-start)/2+start;

    all_pts = np.round(np.arange(start, stop+0.01, step),1) ;  
    [gridx,gridy] = np.meshgrid(all_pts,all_pts);
    all_grid_points = np.array([gridx.ravel(),gridy.ravel()]).T

    return all_grid_points
    
def get_full_grid_maintask():

    # % first I'm defining all possible images that we can use in this task. 
    start = 0;   
    stop = 5;
    step = 0.1;
    center = (stop-start)/2+start;

    all_pts = np.round(np.arange(start, stop+0.01, step),1) ;  
    [gridx,gridy] = np.meshgrid(all_pts,all_pts);
    all_grid_points = np.array([gridx.ravel(),gridy.ravel()]).T

    nsteps_main = 4;    
    start_grid = 0.1;
    stop_grid = 4.9;
    main_pts = np.round(np.linspace(start_grid,stop_grid, nsteps_main),1)

    # % now taking out images at exactly the prototype locations so that we
    # % never use these during task
    proto_pts = [np.round(np.mean(main_pts[0:2]),1), np.round(np.mean(main_pts[2:4]),1)];
    proto_coords = np.array([[proto_pts[1], proto_pts[1]], \
                            [proto_pts[0], proto_pts[1]], \
                            [proto_pts[0], proto_pts[0]], \
                            [proto_pts[1], proto_pts[0]]])
    proto_inds = [np.where(np.all(all_grid_points==proto_coords[ii,:], axis=1))[0][0] \
                  for ii in range(4)]
    inds_remove = np.zeros((all_grid_points.shape[0],), dtype=bool)
    inds_remove[proto_inds] = True
    all_grid_points = all_grid_points[~inds_remove]

    # % Also taking out any images along quadrant boundaries because these
    # % are ambiguous 
    bound_inds = np.any(all_grid_points==center,axis=1)
    all_grid_points = all_grid_points[~bound_inds]
   
    # % now define which quadrant each image lies in. note that this
    # % "quadrant" property is fixed no matter what the task is, the
    # % "category" is a separate property that will change depending on task.
    # % that gets defined later.
    all_quadrant = np.zeros((np.shape(all_grid_points)[0],0))
    all_quadrant[(all_grid_points[:,0]>center) & (all_grid_points[:,1]>center)] = 1;
    all_quadrant[(all_grid_points[:,0]<center) & (all_grid_points[:,1]>center)] = 2;
    all_quadrant[(all_grid_points[:,0]<center) & (all_grid_points[:,1]<center)] = 3;
    all_quadrant[(all_grid_points[:,0]>center) & (all_grid_points[:,1]<center)] = 4;

    # % Next, for each point in the full grid, define the difficulty level 
    # % based on distance from the boundary. This depends on the active task
    # % because it's relative to the active boundary.

    dist_from_bound1 = np.round(np.abs(all_grid_points[:,0]-center),1);
    dist_from_bound2 = np.round(np.abs(all_grid_points[:,1]-center),1);
    dist_from_bound3 = np.round(np.min(np.abs(all_grid_points-center),axis=1),1);

    # % bin these values into 13 "difficulty levels"
    [undist1, dist_groups1] = np.unique(np.round(dist_from_bound1,1),return_inverse=True); 
    [undist2, dist_groups2] = np.unique(np.round(dist_from_bound2,1),return_inverse=True); 
    [undist3, dist_groups3] = np.unique(np.round(dist_from_bound3,1),return_inverse=True); 

    # % define the start of each bin
    bin_edges = np.flip(np.array(list(np.arange(0.1, 0.901, 0.1)) + list(np.arange(1.2, 2.101, 0.3))))
    bin_edges = np.round(bin_edges,1)
   
    dist_groups_binned1 = np.zeros(np.shape(dist_groups1));
    dist_groups_binned2 = np.zeros(np.shape(dist_groups2));
    dist_groups_binned3 = np.zeros(np.shape(dist_groups2));
    for bb in range(len(bin_edges)):
        if bb>0:
            inds = (dist_from_bound1>=bin_edges[bb]) & (dist_from_bound1<bin_edges[bb-1])
        else:
            inds = dist_from_bound1>=bin_edges[bb]
        dist_groups_binned1[inds] = bb;
        if bb>0:
            inds = (dist_from_bound2>=bin_edges[bb]) & (dist_from_bound2<bin_edges[bb-1])
        else:
            inds = dist_from_bound2>=bin_edges[bb]
        dist_groups_binned2[inds] = bb;
        if bb>0:
            inds = (dist_from_bound3>=bin_edges[bb]) & (dist_from_bound3<bin_edges[bb-1])
        else:
            inds = dist_from_bound3>=bin_edges[bb]
        dist_groups_binned3[inds] = bb;

    n_trials_variable = 16
    bins, nperbin = np.unique(dist_groups_binned1, return_counts=True) 
    assert(all(nperbin>=n_trials_variable));
    bins, nperbin = np.unique(dist_groups_binned2, return_counts=True) 
    assert(all(nperbin>=n_trials_variable));
    bins, nperbin = np.unique(dist_groups_binned3, return_counts=True) 
    assert(all(nperbin>=n_trials_variable));
    
    
    return all_grid_points, [dist_groups_binned1, dist_groups_binned2, dist_groups_binned3]


def get_neighbors_repeattask(pt, difficulty_level):

    main_grid_points = get_main_grid()
    main_pts = np.unique(main_grid_points[:,0])
    max_dist = np.round(np.diff(main_pts[0:2])[0]/2, 1)
    step = 0.1;
    start = 0; stop = 5;

    difficulty_distance_bins = np.flipud(np.array([np.arange(step, max_dist-step+0.01, step), \
                                                   np.arange(2*step, max_dist+0.01, step)]).T - 0.05)

    dist_range = difficulty_distance_bins[difficulty_level, :]

    # % for this point, define all the points that neighbor it
    
    [nx,ny] = np.meshgrid(np.arange(pt[0]-max_dist+step, pt[0]+max_dist-step+0.01, step), 
                          np.arange(pt[1]-max_dist+step, pt[1]+max_dist-step+0.01, step))

    nx = np.round(nx,1).ravel();
    ny = np.round(ny,1).ravel();
    
    all_neighbors = np.array([nx,ny]).T;

    # % now filter out only the ones within the desired "distance" from
    # % the point of interest.
    in_bounds = (nx<=stop) & (ny<=stop) & (nx>=start) & (ny>=start);
    rad = np.sqrt(np.sum((all_neighbors-np.tile(pt[:,None].T, [all_neighbors.shape[0],1]))**2,axis=1));
    
    neighbors = all_neighbors[in_bounds & (rad>=dist_range[0]) & (rad<dist_range[1]),:]

    return neighbors
