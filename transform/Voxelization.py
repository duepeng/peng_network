from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import cumm.tensorview as tv
import numpy as np

class Voxelization():
    def __init__(self, 
                 voxel_size,
                 pts_range,
                 pt_dim,
                 max_pts_per_voxel, # if the pts num in voxel is larger than max_pts_per_voxel, the other pts will be discarded.
                 max_voxel_num):
        
        self.voxel_generator = VoxelGenerator(
             vsize_xyz = voxel_size ,
             coors_range_xyz = pts_range, #np.array(pts_range) ,
             num_point_features = pt_dim ,
             max_num_points_per_voxel = max_pts_per_voxel ,
             max_num_voxels = max_voxel_num
        )

    def generate(self, pts):
        voxel_output = self.voxel_generator.point_to_voxel(tv.from_numpy(pts))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        # voxels: [voxel_num, max_pts_per_voxel, pt_dim]
        # if the pts num in voxel is less than max_pts_per_voxel, use 0 to padd.
        return voxels, coordinates, num_points


if __name__ == '__main__':
    # gen fake pts
    pts_xy = np.random.uniform(low=-50,high=50,size=(10000, 2))
    pts_z = np.random.uniform(low=-5,high=5,size=(10000, 1))
    pts = np.hstack([pts_xy, pts_z], )

    # paras
    voxel_size = [0.075, 0.075, 0.2]
    pts_range=  [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    pt_dim = 3
    max_pts_per_voxel = 10
    max_voxel_num = 120000

    
    

    voxelization = Voxelization(voxel_size,
                                pts_range,
                                pt_dim,
                                max_pts_per_voxel,
                                max_voxel_num)
    
    voxels, coordinates, num_points = voxelization.generate(pts)
    pass