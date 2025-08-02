import jittor as jt
import sys
from datetime import datetime
import numpy as np
import random

def split_mesh_and_gaussian(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,v_origin_num):
# in:new_vertex1:[N,4,3],new_v_index[N,4,3],new_v:[N,3,3],v_origin_num:原来顶点的个数
# out:[N,4,3]
    
    a = new_vertex1[:,0,:].clone()
    b = new_vertex2[:,0,:].clone()
    c = new_vertex3[:,0,:].clone()
    new_vertex1[:,0,:] = a
    new_vertex1[:,1,:] = (a+b)/2
    new_vertex1[:,2,:] = (a+c)/2
    new_vertex1[:,3,:] = (a+b)/2
    new_vertex2[:,0,:] = (a+b)/2
    new_vertex2[:,1,:] = b
    new_vertex2[:,2,:] = (c+b)/2
    new_vertex2[:,3,:] = (b+c)/2
    new_vertex3[:,0,:] = (a+c)/2
    new_vertex3[:,1,:] = (c+b)/2
    new_vertex3[:,2,:] = c
    new_vertex3[:,3,:] = (a+c)/2

    new_v[:,0,:] = (a+b)/2
    new_v[:,1,:] = (a+c)/2
    new_v[:,2,:] = (b+c)/2

    tmp = jt.arange(new_v.shape[0]*3).view(new_v.shape[0],3)
    

    new_v_index[:,0,1] = tmp[:,0] + v_origin_num
    new_v_index[:,0,2] = tmp[:,1] + v_origin_num
    new_v_index[:,1,0] = tmp[:,0] + v_origin_num
    new_v_index[:,1,2] = tmp[:,2] + v_origin_num
    new_v_index[:,2,0] = tmp[:,1] + v_origin_num
    new_v_index[:,2,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,0] = tmp[:,0] + v_origin_num
    new_v_index[:,3,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,2] = tmp[:,1] + v_origin_num

    return new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index

def split_mesh_and_gaussian_pro(new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index,v_origin_num):
# in:new_vertex1:[N,5,3],new_v_index[N,5,3],new_v:[N,3,3],v_origin_num:原来顶点的个数
# out:[N,5,3]
    
    a = new_vertex1[:,0,:].clone()
    b = new_vertex2[:,0,:].clone()
    c = new_vertex3[:,0,:].clone()
    new_vertex1[:,0,:] = a
    new_vertex1[:,1,:] = (a+b)/2
    new_vertex1[:,2,:] = (a+c)/2
    new_vertex1[:,3,:] = (a+b)/2
    new_vertex2[:,0,:] = (a+b)/2
    new_vertex2[:,1,:] = b
    new_vertex2[:,2,:] = (c+b)/2
    new_vertex2[:,3,:] = (b+c)/2
    new_vertex3[:,0,:] = (a+c)/2
    new_vertex3[:,1,:] = (c+b)/2
    new_vertex3[:,2,:] = c
    new_vertex3[:,3,:] = (a+c)/2
    new_vertex1[:,4,:] = a
    new_vertex2[:,4,:] = b
    new_vertex3[:,4,:] = c

    new_v[:,0,:] = (a+b)/2
    new_v[:,1,:] = (a+c)/2
    new_v[:,2,:] = (b+c)/2

    tmp = jt.arange(new_v.shape[0]*3).view(new_v.shape[0],3)
    

    new_v_index[:,0,1] = tmp[:,0] + v_origin_num
    new_v_index[:,0,2] = tmp[:,1] + v_origin_num
    new_v_index[:,1,0] = tmp[:,0] + v_origin_num
    new_v_index[:,1,2] = tmp[:,2] + v_origin_num
    new_v_index[:,2,0] = tmp[:,1] + v_origin_num
    new_v_index[:,2,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,0] = tmp[:,0] + v_origin_num
    new_v_index[:,3,1] = tmp[:,2] + v_origin_num
    new_v_index[:,3,2] = tmp[:,1] + v_origin_num

    return new_vertex1,new_vertex2,new_vertex3,new_v,new_v_index

def get_barycentric_coordinate(gaussians,p1,p2,p3):
    e1 = gaussians-p1
    e2 = gaussians-p2
    e3 = gaussians-p3
    s1 = np.cross(e2,e3)
    s1 = np.linalg.norm(s1,axis=1)
    s2 = np.cross(e1,e3)
    s2 = np.linalg.norm(s2,axis=1)
    s3 = np.cross(e1,e2)
    s3 = np.linalg.norm(s3,axis=1)
    s1 = np.expand_dims(s1,axis=1)
    s2 = np.expand_dims(s2,axis=1)
    s3 = np.expand_dims(s3,axis=1)
    s = s1+s2+s3
    coord = np.concatenate([s1/s,s2/s,s3/s],axis =1)
    return coord