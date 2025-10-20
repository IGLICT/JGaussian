cd ops

cd ACAP
unzip pyACAPv1.zip
chmod +x install_OpenMesh.sh
install_OpenMesh.sh
python setup.py install
cd ..

# 3DGS
cd diff_gaussian_rasterization
cmake .
make -j
cd ..

# 2DGS 
cd diff_surfel_rasterization
cmake .
make -j
cd ..

# Mip-splatting
cd mip_diff_gaussian_rasterizater
cmake .
make -j
cd ..

cd simple_knn
cmake .
make -j
cd ..

cd NVDIFFREC
cd texture
cmake .
make -j
cd ..
cd renderutils
cmake .
make -j
cd ../../..

cd application/gsheadrelighting/jittorutils
cmake .
make -j
cd ../../..
