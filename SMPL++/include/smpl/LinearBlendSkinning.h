/* ========================================================================= *
 *                                                                           *
 *                                 SMPL++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - SMPL: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original SMPL model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  CLASS LinearBlendSkinning DECLARATIONS
//
//=============================================================================

#ifndef LINEAR_BLEND_SKINNING_H
#define LINEAR_BLEND_SKINNING_H

//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

#include <torch/torch.h>
#include <fstream>
#include <iostream>
//using namespace std;
//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {


//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS DEFINITIONS =====================================================

/**
 * DESCRIPTIONS:
 * 
 *      Fourth of the four modules in SMPL pipeline.
 * 
 *      This class finally implement pose change of the model. We will apply
 *      the most popular skinning method - linear blend skinning - to all
 *      vertices. As its definition, linear blend skinning combines
 *      transformations of different near bones together to get the finally
 *      transformation. Indeed, this one doesn't guarantee a rigid
 *      transformation any more. Hopefully, we may use more sophistatic
 *      skinning model like dual quaternion skinning to fine tune the pose.
 * 
 * 
 * INHERITANCES:
 * 
 * 
 * ATTRIBUTES:
 * 
 *      - m__device: <private>
 *          Torch device to run the module, could be CPUs or GPUs.
 * 
 *      - m__restShape: <private>
 *          Deformed shape in rest pose, (N, 6890, 3).
 * 
 *      - m__transformation: <private>
 *          World transformation expressed in homogeneous coordinates
 *          after eliminating effects of rest pose, (N, 24, 4, 4).
 * 
 *      - m__weights: <private>
 *          Weights for linear blend skinning, (6890, 24).
 * 
 *      - m__posedVert: <private>
 *           Vertex locations of the new pose, (N, 6890, 3).
 * 
 * METHODS:
 * 
 *      %
 *          Constructor and Destructor
 *      %
 *      - LinearBlendSkinning: <public>
 *          Default constructor.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Constructor to initialize weights and torch device for linear 
 *          blend skinning.
 * 
 *      - LinearBlendSkinning: (overload) <public>
 *          Copy constructor.
 * 
 *      - ~LinearBlendSkinning: <public>
 *          Destructor.
 *      %%
 * 
 *      %
 *          Operators
 *      %
 *      - operator=: <public>
 *          Assignment is used to copy a <LinearBlendSkinning> instantiation.
 *      %%
 * 
 *      %
 *          Getter and Setter
 *      %
 *      - setDevice: <public>
 *          Set the torch device.
 * 
 *      - setRestShape: <public>
 *          Set the deformed shape in rest pose.
 * 
 *      - setWeight: <public>
 *          Set the weights for linear blend skinning.
 * 
 *      - setTransformation: <public>
 *          Set the world transformation.
 * 
 *      - getVertex: <public>
 *          Get vertex locations of the new pose.
 * 
 *      %%
 * 
 *      %
 *          Linear Blend Skinning
 *      %
 *      - skinning: <public>
 *          Do all the skinning stuffs.
 * 
 *      - cart2homo: <private>
 *          Convert Cartesian coordinates to homogeneous coordinates.
 * 
 *      - homo2cart: <private>
 *          Convert homogeneous coordinates to Cartesian coordinates.
 *      %%
 * 
 * 
 */
struct person 
{
public:
	int m_id;
	torch::Tensor m_Rh;
	torch::Tensor m_Th;
	torch::Tensor m_poses;
	torch::Tensor m_shapes;
	person(int id, torch::Tensor Rh, torch::Tensor Th, torch::Tensor poses, torch::Tensor shapes);
	virtual ~person() {};
};

class LinearBlendSkinning final
{

private: // PIRVATE ATTRIBUTES

    torch::Device m__device;

    torch::Tensor m__restShape;
    
    torch::Tensor m__weights;
    torch::Tensor m__posedVert;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES
    torch::Tensor m__transformation;
private: // PRIVATE METHODS

protected: // PROTECTED METHODS

public: // PUBLIC METHODS
    // %% Linear Blend Skinning %%
    torch::Tensor cart2homo(torch::Tensor& cart) noexcept(false);
    torch::Tensor homo2cart(torch::Tensor& homo) noexcept(false);

    // %% Constructor and Destructor %%
    LinearBlendSkinning() noexcept(true);
    LinearBlendSkinning(torch::Tensor &weights, 
        torch::Device &device) noexcept(false);
    LinearBlendSkinning(const LinearBlendSkinning &linearBlendSkinning)
        noexcept(false);
    ~LinearBlendSkinning() noexcept(true);

    // %% Operator %%
    LinearBlendSkinning &operator=(
        const LinearBlendSkinning &linearBlendSkinning) noexcept(false);

    // %% Setter and Getter %%
    void setDevice(const torch::Device &device) noexcept(false);
    void setWeight(const torch::Tensor &weights) noexcept(false);
    void setRestShape(const torch::Tensor &restShape) noexcept(false);
    void setTransformation(
        const torch::Tensor &transformation) noexcept(false);

    torch::Tensor getVertex() noexcept(false);

    // %% Linear Blend Skinning %%
    void skinning() noexcept(false);

    //////////////////////////////////////////////////////////////////////////
    //anzs
    // 
	// def hybrik( betas, global_orient, pose_skeleton, #phis,
	//v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
	//	lbs_weights, dtype = torch.float32, train = False, leaf_thetas = None):
	//	''' Performs Linear Blend Skinning with the given shape and skeleton joints
    //
    /*
	*
		Parameters
		----------
		betas : torch.tensor BxNB
			The tensor of shape parameters
		global_orient : torch.tensor Bx3
			The tensor of global orientation
		pose_skeleton : torch.tensor BxJ*3
			The pose skeleton in (X, Y, Z) format
		phis : torch.tensor BxJx2
			The rotation on bone axis parameters
		v_template torch.tensor BxVx3
			The template mesh that will be deformed
		shapedirs : torch.tensor 1xNB
			The tensor of PCA shape displacements
		posedirs : torch.tensor Px(V * 3)
			The pose PCA coefficients
		J_regressor : torch.tensor JxV
			The regressor array that is used to calculate the joints from
			the position of the vertices
		J_regressor_h36m : torch.tensor 17xV
			The regressor array that is used to calculate the 17 Human3.6M joints from
			the position of the vertices
		parents: torch.tensor J
			The array that describes the kinematic parents for the model
		children: dict
			The dictionary that describes the kinematic chidrens for the model
		lbs_weights: torch.tensor N x V x (J + 1)
			The linear blend skinning weights that represent how much the
			rotation matrix of each part affects each vertex
		dtype: torch.dtype, optional

		Returns
		-------
		verts: torch.tensor BxVx3
			The vertices of the mesh after applying the shape and pose
			displacements.
		joints: torch.tensor BxJx3
			The joints of the model
		rot_mats: torch.tensor BxJx3x3
			The rotation matrics of each joints
	'''

    */
	torch::Tensor blend_shapes(const torch::Tensor& betas, const torch::Tensor& shape_disps);
	torch::Tensor vertices2joints(const torch::Tensor& J_regressor, const torch::Tensor& shape_disps);
// 	def vertices2joints(J_regressor, vertices) :
// 		''' Calculates the 3D joint locations from the vertices
// 
// 		Parameters
// 		----------
// 		J_regressor : torch.tensor JxV
// 		The regressor array that is used to calculate the joints from the
// 		position of the vertices
// 		vertices : torch.tensor BxVx3
// 		The tensor of mesh vertices
// 
// 		Returns
// 		------ -
// 		torch.tensor BxJx3
// 		The location of the joints
// 		'''
// 
// 		return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


	void write_persons(std::vector<person*> persons, std::ofstream& file);
	void write_json(std::ofstream& file, const int id, const torch::Tensor& Rh, const torch::Tensor& Th, const torch::Tensor& poses, const torch::Tensor& shapes);

	void hybrik(const torch::Tensor& torpose_skeleton,
		const torch::Tensor& betas,
		//const torch::Tensor& global_orient,
		//phis,
		const torch::Tensor& v_template,
		const torch::Tensor& shapedirs,
		const torch::Tensor& posedirs,
		const torch::Tensor& J_regressor,
		const torch::Tensor& J_regressor_h36m,
		const torch::Tensor& parents,
		const torch::Tensor& children,
		const torch::Tensor& lbs_weights);// dtype = torch.float32, train = False, leaf_thetas = None)


	torch::Tensor batch_inverse_kinematics_transform(
		const torch::Tensor& pose_skeleton,// global_orient,		
		const torch::Tensor& rest_pose,
		const torch::Tensor& children, 
		const torch::Tensor& parents// dtype = torch.float32, train = False,
		//const torch::Tensor& leaf_thetas = None
	);

	torch::Tensor batch_get_pelvis_orient_svd(
		const torch::Tensor& rel_pose_skeleton, 
		const torch::Tensor& rel_rest_pose,
		const torch::Tensor& parents, 
		const torch::Tensor& children
		//dtype
	);

	torch::Tensor batch_get_3children_orient_svd(
		std::vector<torch::Tensor> rel_pose_skeleton,
		std::vector<torch::Tensor> rel_rest_pose,
		torch::Tensor rot_mat_chain_parent,
		std::vector<int> children_list);
	torch::Tensor rotation_matrix_to_quaternion(torch::Tensor& rotation_matrix, double eps = 0.000001);
	torch::Tensor rotmat_to_quat(torch::Tensor& rotation_matrix);
	torch::Tensor quaternion_to_angle_axis(torch::Tensor& quaternion);

};

//=============================================================================
} // namespace smpl
//=============================================================================
#endif // LINEAR_BLEND_SKINNING_H
//=============================================================================
