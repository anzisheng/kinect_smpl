#include <cam/smplcam.h>

smplcam::smplcam(torch::Device device)
{
	m_smpl = nullptr;


	//anzs ����xyz.npy
	cnpy::NpyArray arr = cnpy::npy_load("data/xyz.npy");
	//std::vector<float> scales;

	//pred_xyz_jts_29 = torch.tensor(pred_xyz_jts_29).cuda()
	//torch::Tensor pred_xyz_jts_29;
	
	m_pred_xyz_jts_29 = torch::from_blob(arr.data<float>(), { 1,29,3 }).to(device);
	std::cout << "xyz:" << m_pred_xyz_jts_29.device() << m_pred_xyz_jts_29<< std::endl; //<< m_pred_xyz_jts_29 << std::endl;
	
	cnpy::NpyArray arrshape = cnpy::npy_load("data/shape.npy");
	m_pred_shape = torch::from_blob(arrshape.data<float>(), { 1,10 }).to(device);
	

	//m_pred_shape;

}

void smplcam::call_forward(/*const torch::Tensor& xyz, const torch::Tensor& shape*/)
{
	
	m_pred_xyz_jts_29 = m_pred_xyz_jts_29 * 2.2;
	std::cout << "pose_skeleton:" << m_pred_xyz_jts_29 << std::endl;
	std::cout << "m_pred_shape:" << m_pred_shape << std::endl;
	m_smpl->hybrik(m_pred_xyz_jts_29, m_pred_shape);



// 	output = self.smpl.hybrik(
// 		pose_skeleton = pred_xyz_jts_29.type(self.smpl_dtype) * 2.2, #self.depth_factor, # unit: meter
// 		#pose_skeleton = pred_xyz_jts_29,
// 		betas = pred_shape.type(self.smpl_dtype),
// 		#phis = None,
// 		#pred_phi.type(self.smpl_dtype),
// 		global_orient = None,
// 		return_verts = True
// 
// 		m_skinner.hybrik();
// 	);

	//m_smpl->hybrik()

	//return oputput;

}