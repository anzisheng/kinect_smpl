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
 //  CLASS LinearBlendSkinning IMPLEMENTATIONS
 //
 //=============================================================================


 //===== EXTERNAL MACROS =======================================================


 //===== INCLUDES ==============================================================

 //----------
#include <chrono>
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/TorchEx.hpp"
#include "smpl/LinearBlendSkinning.h"
#include "torch/script.h"
using namespace torch::indexing;
#include <exception>
#include "cnpy.h"

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
//----------
#define COUT_VAR(x) std::cout << #x"=" << x << std::endl;
#define COUT_ARR(x) std::cout << "---------"#x"---------" << std::endl;\
        std::cout << x << std::endl;\
        std::cout << "---------"#x"---------" << std::endl;

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl
{

    //===== INTERNAL MACROS =======================================================


    //===== INTERNAL FORWARD DECLARATIONS =========================================


    //===== CLASS IMPLEMENTATIONS =================================================

    /**LinearBlendSkinning
     *
     * Brief
     * ----------
     *
     *      Default constructor.
     *
     * Arguments
     * ----------
     *
     *
     * Return
     * ----------
     *
     *
     */
    LinearBlendSkinning::LinearBlendSkinning() noexcept(true) :
        m__device(torch::kCPU),
        m__restShape(),
        m__transformation(),
        m__weights(),
        m__posedVert()
    {
    }

    /**LinearBlendSkinning (overload)
     *
     * Brief
     * ----------
     *
     *      Constructor to initialize weights for linear blend skinning.
     *
     * Arguments
     * ----------
     *
     *      @weights: - Tensor -
     *          Weights for linear blend skinning, (6890, 24).
     *
     *      @device: - Device -
     *          Torch device to run the module, CPUs or GPUs.
     *
     * Return
     * ----------
     *
     *
     */
    LinearBlendSkinning::LinearBlendSkinning(torch::Tensor& weights,
        torch::Device& device)
        noexcept(false) :
        m__device(torch::kCPU)
    {
        if (device.has_index())
        {
            m__device = device;
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to fetch device index!");
        }

        if (weights.sizes() ==
            torch::IntArrayRef({ VERTEX_NUM, JOINT_NUM }))
        {
            m__weights = weights.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to initialize linear blend weights!");
        }
    }

    /**LinearBlendSkinning (overload)
     *
     * Brief
     * ----------
     *
     *      Copy constructor.
     *
     * Arguments
     * ----------
     *
     *      @linearBlendSkinning: - const LinearBlendSkinning & -
     *          The <LinearBlendSkinning> instantiation to copy with.
     *
     * Return
     * ----------
     *
     *
     */
    LinearBlendSkinning::LinearBlendSkinning(
        const LinearBlendSkinning& linearBlendSkinning) noexcept(false) :
        m__device(torch::kCPU),
        m__posedVert()
    {
        try
        {
            *this = linearBlendSkinning;
        }
        catch (std::exception& e)
        {
            throw;
        }
    }

    /**~LinearBlendSkinning
     *
     * Brief
     * ----------
     *
     *      Destructor.
     *
     * Arguments
     * ----------
     *
     *
     * Return
     * ----------
     *
     *
     */
    LinearBlendSkinning::~LinearBlendSkinning() noexcept(true)
    {
    }

    /**operator=
     *
     * Brief
     * ----------
     *
     *      Assignment is used to copy a <LinearBlendSkinning> instantiation.
     *
     * Arguments
     * ----------
     *
     *      @linearBlendSkinning: - Tensor -
     *          The <LinearBlendSkinning> instantiation to copy with.
     *
     * Return
     * ----------
     *
     *      @*this: - LinearBlendSkinning & -
     *          Current instantiation.
     *
     */
    LinearBlendSkinning& LinearBlendSkinning::operator=(
        const LinearBlendSkinning& linearBlendSkinning) noexcept(false)
    {
        //
        // hard copy
        //
        if (linearBlendSkinning.m__device.has_index())
        {
            m__device = linearBlendSkinning.m__device;
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to fetch device index!");
        }

        if (linearBlendSkinning.m__restShape.sizes() ==
            torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 3 }))
        {
            m__restShape = linearBlendSkinning.m__restShape.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to copy deformed shape in rest pose!");
        }

        if (linearBlendSkinning.m__transformation.sizes() ==
            torch::IntArrayRef({ BATCH_SIZE, JOINT_NUM, 4, 4 }))
        {
            m__transformation = linearBlendSkinning.m__transformation.clone().to(
                m__device);

        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to copy world transformation!");
        }

        if (linearBlendSkinning.m__weights.sizes() ==
            torch::IntArrayRef({ VERTEX_NUM, JOINT_NUM }))
        {
            m__weights = linearBlendSkinning.m__weights.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to copy linear blend weights!");
        }

        //
        // soft copy
        //
        if (linearBlendSkinning.m__posedVert.sizes() ==
            torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 3 }))
        {
            m__posedVert = linearBlendSkinning.m__posedVert.clone().to(m__device);
        }

        return *this;
    }

    /**setDevice
     *
     * Brief
     * ----------
     *
     *      Set the torch device.
     *
     * Arguments
     * ----------
     *
     *      @device: - const Device & -
     *          The torch device to be used.
     *
     * Return
     * ----------
     *
     */
    void LinearBlendSkinning::setDevice(const torch::Device& device) noexcept(false)
    {
        if (device.has_index())
        {
            m__device = device;
        }
        else
        {
            throw smpl_error("LinearBlendSkinning", "Failed to fetch device index!");
        }

        return;
    }

    /**setRestShape
     *
     * Brief
     * ----------
     *
     *      Set the deformed shape in rest pose.
     *
     * Arguments
     * ----------
     *
     *      @restShape: - Tensor -
     *          Deformed shape in rest pose, (N, 6890, 3).
     *
     * Return
     * ----------
     *
     *
     */
    void LinearBlendSkinning::setRestShape(
        const torch::Tensor& restShape) noexcept(false)
    {
        if (restShape.sizes() ==
            torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 3 }))
        {
            m__restShape = restShape.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to set deformed shape in rest pose!");
        }

        return;
    }

    /**setWeight
     *
     * Brief
     * ----------
     *
     *      Set the weights for linear blend skinning.
     *
     * Arguments
     * ----------
     *
     *      weights: - Tensor -
     *          Weights for linear blend skinning, (6890, 24).
     *
     * Return
     * ----------
     *
     *
     */
    void LinearBlendSkinning::setWeight(
        const torch::Tensor& weights) noexcept(false)
    {
        if (weights.sizes() ==
            torch::IntArrayRef({ VERTEX_NUM, JOINT_NUM }))
        {
            m__weights = weights.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to set linear blend weights!");
        }

        return;
    }

    /**setTransformation
     *
     * Brief
     * ----------
     *
     *      Set the world transformation.
     *
     * Arguments
     * ----------
     *
     *      @transformation: - Tensor -
     *          World transformation expressed in homogeneous coordinates
     *          after eliminating effects of rest pose, (N, 24, 4, 4).
     *
     * Return
     * ----------
     *
     *
     */
    void LinearBlendSkinning::setTransformation(
        const torch::Tensor& transformation) noexcept(false)
    {
        if (transformation.sizes() ==
            torch::IntArrayRef({ BATCH_SIZE, JOINT_NUM, 4, 4 }))
        {
            m__transformation = transformation.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSkinning",
                "Failed to set world transformation!");
        }

        return;
    }

    /**getVertex
     *
     * Brief
     * ----------
     *
     *      Get vertex locations of the new pose.
     *
     * Arguments
     * ----------
     *
     *
     * Return
     * ----------
     *
     *      @vertices: - Tensor -
     *          Vertex locations of the new pose, (N, 6890, 3).
     *
     */
    torch::Tensor LinearBlendSkinning::getVertex() noexcept(false)
    {
       //COUT_ARR(m__posedVert)
        torch::Tensor vertices;
        if (m__posedVert.sizes() == torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 3 }))
        {
            vertices = m__posedVert.clone().to(m__device);
        }
        else
        {
            throw smpl_error("LinearBlendSknning",
                "Failed to get vertices of new pose!");
        }

        return vertices;
    }

    /**skinning
     *
     * Brief
     * ----------
     *
     *      Do all the skinning stuffs.
     *
     * Arguments
     * ----------
     *
     *
     * Return
     * ----------
     *
     *
     */
    void LinearBlendSkinning::skinning() noexcept(false)
    {
        //
        // Cartesian coordinates to homogeneous coordinates
        //
        torch::Tensor restShapeHomo;
        try
        {
            restShapeHomo = cart2homo(m__restShape);// (N, 6890, 4)
        }
        catch (std::exception& e)
        {
            throw;
        }

        //
        // linear blend skinning
        //
        torch::Tensor coefficients = torch::tensordot(
            m__weights, m__transformation, { 1 }, { 1 });// (6890, N, 4, 4)
        coefficients = torch::transpose(coefficients, 0, 1);// (N, 6890, 4, 4)
        restShapeHomo = torch::unsqueeze(restShapeHomo, 3);// (N, 6890, 4, 1)
        torch::Tensor verticesHomo = torch::matmul(
            coefficients, restShapeHomo);// (N, 6890, 4, 1)
        verticesHomo = torch::squeeze(verticesHomo, 3);// (N, 6890, 4)

        //
        // homogeneous coordinates to Cartesian coordinates
        //
        try
        {
            m__posedVert = homo2cart(verticesHomo);
        }
        catch (std::exception& e)
        {
            throw;
        }

        return;
    }

    /**cart2homo
     *
     * Brief
     * ----------
     *
     *      Convert Cartesian coordinates to homogeneous coordinates.
     *
     * Argument
     * ----------
     *
     *      @cart: - Tensor -
     *          Vectors in Cartesian coordinates, (N, 6890, 3).
     *
     * Return
     * ----------
     *
     *      @homo: - Tensor -
     *          Vectors in homogeneous coordinates, (N, 6890, 4).
     *
     */
    torch::Tensor LinearBlendSkinning::cart2homo(torch::Tensor& cart)
        noexcept(false)
    {
        if (cart.sizes() !=
            torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 3 }))
        {
            throw smpl_error("LinearBlendSkinning",
                "Cannot convert Cartesian coordinates to homogeneous one!");
        }

        torch::Tensor ones = torch::ones(
            { BATCH_SIZE, VERTEX_NUM, 1 }, m__device);// (N, 6890, 1)
        torch::Tensor homo = torch::cat({ cart, ones }, 2);// (N, 6890, 4)

        return homo;
    }

    /**homo2cart
     *
     * Brief
     * ----------
     *
     *      Convert Cartesian coordinates to homogeneous coordinates.
     *
     * Argument
     * ----------
     *
     *      @homo: - Tensor -
     *          Vectors in homogeneous coordinates, (N, 6890, 4).
     *
     * Return
     * ----------
     *
     *      @cart: - Tensor -
     *          Vectors in Cartesian coordinates, (N, 6890, 3).
     *
     */
    torch::Tensor LinearBlendSkinning::homo2cart(torch::Tensor& homo)
        noexcept(false)
    {
        if (homo.sizes() !=
            torch::IntArrayRef({ BATCH_SIZE, VERTEX_NUM, 4 }))
        {
            throw smpl_error("LinearBlendSkinning",
                "Cannot convert homogeneous coordinates to Cartesian ones!");
        }

        torch::Tensor homoW = TorchEx::indexing(homo,
            torch::IntList(),
            torch::IntList(),
            torch::IntList({ 3 }));// (N, 6890)
        homoW = torch::unsqueeze(homoW, 2);// (N, 6890, 1)
        torch::Tensor homoUnit = homo / homoW;// (N, 6890, 4)
        torch::Tensor cart = TorchEx::indexing(homoUnit,
            torch::IntList(),
            torch::IntList(),
            torch::IntList({ 0, 3 }));// (N, 6890, 3)

        return cart;
    }

    torch::Tensor LinearBlendSkinning::blend_shapes(const torch::Tensor& betas, const torch::Tensor& shape_disps)
    {
		/* Calculates the per vertex displacement due to the blend shapes


			Parameters
			----------
			betas : torch.tensor Bx(num_betas)
			Blend shape coefficients
			shape_disps : torch.tensor Vx3x(num_betas)
			Blend shapes

			Returns
			------ -
			torch.tensor BxVx3
			The per - vertex displacement due to shape deformation
			*/

			//Displacement[b, m, k] = sum_{ l } betas[b, l] * shape_disps[m, k, l]
			// i.e.Multiply each shape displacement by its corresponding beta and
			// then sum them.
        if(SHOWOUT)
        {
            std::cout << "betas shape:" << betas<< std::endl;
            std::cout << "shape_disps[-1,:,-1]:" << shape_disps.index({-1, Slice(),-1}) << std::endl;
        }
        //print("shape_disps[-1,:,-1]", shape_disps[-1, :, -1])
        //std::cout << "blend_shape shape:" << blend_shape.sizes() << std::endl;

        torch::Tensor blend_shape;        
        try
        {
            //at::TensorList t = { betas, shape_disps };
            blend_shape = torch::einsum("bl, mkl->bmk", { betas, shape_disps });
            
        }
		catch (std::exception& e)
		{
            std::cout << e.what() << std::endl;
			throw;
		}
        //print("blend_shape[：,-1,:]", blend_shape[:, -1, :])
        if (SHOWOUT)
        {
            std::cout << "blend_shape shape[：,-1,:]:" << blend_shape.index({ Slice(), -1, Slice() }) << std::endl;
        }
        return blend_shape;

    }

    torch::Tensor LinearBlendSkinning::batch_inverse_kinematics_transform(
        const torch::Tensor& pose_skeleton,// global_orient,		
        const torch::Tensor& rest_pose,
        const torch::Tensor& children,
        const torch::Tensor& parents// dtype = torch.float32, train = False,
        //const torch::Tensor& leaf_thetas = None
    )
    {
        /*
		*     """
	Applies a batch of inverse kinematics transfoirm to the joints

	Parameters
	----------
	pose_skeleton : torch.tensor BxNx3
		Locations of estimated pose skeleton.
	global_orient : torch.tensor Bx1x3x3
		Tensor of global rotation matrices
	phis : torch.tensor BxNx2
		The rotation on bone axis parameters
	rest_pose : torch.tensor Bx(N+1)x3
		Locations of rest_pose. (Template Pose)
	children: dict
		The dictionary that describes the kinematic chidrens for the model
	parents : torch.tensor Bx(N+1)
		The kinematic tree of each object
	dtype : torch.dtype, optional:
		The data type of the created tensors, the default is torch.float32

	Returns
	-------
	rot_mats: torch.tensor Bx(N+1)x3x3
		The rotation matrics of each joints
	rel_transforms : torch.tensor Bx(N+1)x4x4
		The relative (with respect to the root joint) rigid transformations
		for all the joints
	"""

        */
        batch_size = pose_skeleton.size(0);
        torch::Device device = this->m__device;
        torch::Tensor rel_rest_pose = rest_pose.clone();
        if (SHOWOUT)
        {
            std::cout << "rel_rest_pose :" << rel_rest_pose.sizes() << std::endl;
        }

        //rel_rest_pose[:, 1 : ] -= rest_pose[:, parents[1:]].clone();

 		std::vector<int> parents_exclude_0_v = { 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11 };
        torch::Tensor parents_exclude_0 = torch::tensor(parents_exclude_0_v);// .data(), { 29 }).toType(torch::kInt8);
        parents_exclude_0 = parents_exclude_0.to(m__device);
        if (SHOWOUT)
        {
            std::cout << "parents_exclude_0:" << parents_exclude_0.sizes() << parents_exclude_0.device() << parents_exclude_0 << std::endl;
        }

        //torch::Tensor par = parents.index({ Slice(1,None) });                
        //std::cout << "par:" << par.sizes() << par << std::endl;

        //torch::Tensor b = torch::tensor({ 0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2 });
//         torch::Tensor temp = rest_pose.index({ Slice(), {parents_exclude_0}});
//         std::cout << "temp :" << temp.sizes() << std::endl;

        //rel_rest_pose.index({ Slice(),Slice(1),Slice() }) -= rest_pose.index({ Slice(), parents.index({Slice(1,None)}) });// .clone();
        // 
        rel_rest_pose.index({ Slice(),Slice(1),Slice() }) -= rest_pose.index({ Slice(), {parents_exclude_0} });
        if (SHOWOUT)
        {
            std::cout << "rel_rest_pose:" << rel_rest_pose.sizes() << rel_rest_pose << std::endl;
            std::cout << "rel_rest_pose.index({ Slice(),Slice(1),Slice() }):" << rel_rest_pose.index({ Slice(),Slice(1),Slice() }).sizes() << rel_rest_pose.index({ Slice(),Slice(1),Slice() }) << std::endl;
        }

        rel_rest_pose = torch::unsqueeze(rel_rest_pose, -1);
        if (SHOWOUT)
        {
            std::cout << "rel_rest_pose:" << rel_rest_pose.sizes() << std::endl; //torch.Size([1, 29, 3, 1])
        }


		//# rotate the T pose
        torch::Tensor rotate_rest_pose = torch::zeros_like(rel_rest_pose);
        if (SHOWOUT)
        {
            std::cout << "rotate_rest_pose:" << rotate_rest_pose.sizes() << std::endl;
        }

		//# set up the root
        //rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]
        rotate_rest_pose.index({ Slice(), 0 }) = rel_rest_pose.index({ Slice(),0 });
        if (SHOWOUT)
        {
            std::cout << "rotate_rest_pose:" << rotate_rest_pose.sizes() << std::endl;
        }


// 		rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim = -1).detach()
//		rel_pose_skeleton[:, 1 : ] = rel_pose_skeleton[:, 1 : ] - rel_pose_skeleton[:, parents[1:]].clone()
// 		rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        //torch::Tensor b2 = torch::tensor({ 0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2 });

        torch::Tensor rel_pose_skeleton = torch::unsqueeze(pose_skeleton.clone(), -1).detach();
		torch::Tensor temp = torch::squeeze(rel_pose_skeleton);
        if (SHOWOUT)
        {
            std::cout << "temp:" << temp.sizes() << temp << std::endl;
        }

        try 
        {
            rel_pose_skeleton.index({ Slice(),Slice(1,None) }) = rel_pose_skeleton.index({ Slice(),Slice(1,None) }) - rel_pose_skeleton.index({ Slice(),{parents_exclude_0} }).clone();
        }
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			throw;
		}

        rel_pose_skeleton.index({ Slice(),0 }) = rel_rest_pose.index({ Slice(),0 });
        if (SHOWOUT)
        {
            std::cout << "rel_pose_skeleton:" << rel_pose_skeleton.sizes() << rel_pose_skeleton << std::endl;
        }

        /*torch::Tensor*/ temp = torch::squeeze(rel_pose_skeleton);
        if (SHOWOUT)
        {
            std::cout << "temp:" << temp.sizes() << temp << std::endl;
        }


// 		# the predicted final pose
// 			final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim = -1)
// 			final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0 : 1] + rel_rest_pose[:, 0 : 1]
        
        torch::Tensor final_pose_skeleton = torch::unsqueeze(pose_skeleton.clone(), -1);
        try
        {
            final_pose_skeleton = final_pose_skeleton - final_pose_skeleton.index({ Slice(), Slice(0,1) }) + rel_rest_pose.index({ Slice(), Slice(0,1) });//
        }
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			throw;
		}
        if (SHOWOUT)
        {
            std::cout << "final_pose_skeleton: " << final_pose_skeleton << std::endl;
        }

		temp = torch::squeeze(final_pose_skeleton);
        if (SHOWOUT)
        {
            std::cout << "temp[[3]]:" << temp.sizes() << temp.index({ 3,Slice() }) << std::endl;
            std::cout << "temp[[13]]:" << temp.sizes() << temp.index({ 13,Slice() }) << std::endl;
            std::cout << "temp[[26]]:" << temp.sizes() << temp.index({ 26,Slice() }) << std::endl;
        }


        rel_rest_pose = rel_rest_pose;
        //rel_pose_skeleton = rel_pose_skeleton;
        final_pose_skeleton = final_pose_skeleton;
        rotate_rest_pose = rotate_rest_pose;


		//count the number of children for each joint
// 			child_indices = []
// 			for j in range(parents.shape[0]) :
// 				child = []
// 				for i in range(1, parents.shape[0]) :
// 					if parents[i] == j and i not in child :
//             child.append(i)
//                 child_indices.append(child)

            std::vector<std::vector<int>> child_indices;
            for (int j = 0; j < 29; j++)
            {
                std::vector<int> child;
                for (int i = 1; i < 29; i++)
                {
                    int nCount = std::count(child.begin(), child.end(), i);

                    if ((parents.index({i}).item() == j) && (nCount <= 0))
                    {
                        child.push_back(i);
                    }
                }
                child_indices.push_back(child);
            }

        /*
		global_orient_mat = batch_get_pelvis_orient_svd(
		rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
        */
            if (SHOWOUT)
            {
                std::cout << "parents :" << parents << std::endl;
                std::cout << "children :" << children << std::endl;
            }

        torch::Tensor global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children);
        //global_orient_mat = global_orient_mat.to(torch::kFloat64);
        if (SHOWOUT)
        {
            std::cout << "torglobal_orient_mat:" << global_orient_mat.sizes() << global_orient_mat << std::endl;
        }

        
        
        
        
        //备注的代码

        //In C++ you can create a std::vector<Tensor>& tensors and use torch::stack(tensors) instead.

//         std::vector<torch::Tensor> tensors;
//         tensors.push_back(torglobal_orient_mat);
//         tensors.push_back(torglobal_orient_mat);
//         tensors.push_back(torglobal_orient_mat);

//         torch::Tensor rot_mats = torch::stack(tensors, 1);
//         std::cout << "rot_mats:" << rot_mats.sizes() << rot_mats << std::endl;
		   //rot_mat_chain = [global_orient_mat]
		   //rot_mat_local = [global_orient_mat]


        std::vector<torch::Tensor> rot_mat_chain;
        std::vector<torch::Tensor> rot_mat_local;

        rot_mat_chain.push_back(global_orient_mat.to(torch::kFloat64));
        rot_mat_local.push_back(global_orient_mat);


        //for i in range(1, parents.shape[0]) :
        for (int i = 1; i < parents.size(0); i++)
        {
            /*
		if children[i] == -1:
			# leaf nodes
			if leaf_thetas is not None:
				rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
				leaf_cnt += 1

				rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
					rot_mat_chain[parents[i]],
					rel_rest_pose[:, i]
				)

				rot_mat_chain.append(torch.matmul(
					rot_mat_chain[parents[i]],
					rot_mat))
				rot_mat_local.append(rot_mat)
		elif len(child_indices[i]) == 3: #children[i] == -3:

            */
            //std::cout << parents.size(0) << children.index({ i }).item();
            if(children.index({ i }).item() == -1)
            {

            }
            else if (child_indices[i].size() == 3)//: #children[i] == -3)
            {
                //只有当i =9,时child_indices[9]才是3，此处硬编码，此时parents[i] 为6
				// three children
// 				rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
// 					rot_mat_chain[parents[i]],
// 					rel_rest_pose[:, i]
// 				)
                if (SHOWOUT)
                {
                    std::cout << "rotate_rest_pose[:,0]:" << rotate_rest_pose.sizes() << rotate_rest_pose.index({ Slice(),0,Slice(),Slice() }) << std::endl;
                }
                rotate_rest_pose = rotate_rest_pose.to(torch::kFloat64);
                if (SHOWOUT)
                {
                    std::cout << "rot_mat_chain[6]:" << rot_mat_chain[6].type() << std::endl;
                }
                rel_rest_pose = rel_rest_pose.to(torch::kFloat64);
                if (SHOWOUT)
                {
                    std::cout << "rel_rest_pose.index({ Slice(),9 }):" << rel_rest_pose.index({ Slice(),9 }).type() << std::endl;
                }

                //此步等等i = 9,再测试，因为到时rot_mat_chain的元素个数才够[6]。
                try 
                {
                    rotate_rest_pose.index({ Slice(),i }) = rotate_rest_pose.index({ Slice(), 6 }) + torch::matmul(rot_mat_chain[6], rel_rest_pose.index({ Slice(),9 }));// [:, i] );// [, parents[i]]
                }
				catch (std::exception& e)
				{
					std::cout << e.what() << std::endl;
					throw;
				}
                if (SHOWOUT)
                {
                    std::cout << "rotate_rest_pose.index({ Slice(),i }):" << rotate_rest_pose.index({ Slice(),i }).sizes() << rotate_rest_pose.index({ Slice(),i }) << std::endl;
                }
                

                /*
				*spine_child = []
			for c in range(1, parents.shape[0]):
				if parents[c] == i and c not in spine_child:
					spine_child.append(c)

                */
                std::vector<int> spine_child = { 12, 13, 14 };

// 				children_final_loc = []
// 				children_rest_loc = []

                std::vector<torch::Tensor> children_final_loc;
                std::vector<torch::Tensor> children_rest_loc;

// 				for c in spine_child :
// 				temp = final_pose_skeleton[:, c] - rotate_rest_pose[:, i]
// 					children_final_loc.append(temp) 
// 					children_rest_loc.append(rel_rest_pose[:, c].clone())

                for (std::vector<int>::iterator iter = spine_child.begin(); iter != spine_child.end(); iter++)
                {
                    int c = *iter;

                    torch::Tensor temp = final_pose_skeleton.index({ Slice(), c }) - rotate_rest_pose.index({ Slice(), i });
                    children_final_loc.push_back(temp);// append()
                    children_rest_loc.push_back(rel_rest_pose.index({ Slice(), c }).clone()); // [:, c] .clone())

                }

                int in = parents.index({ i }).to(torch::kInt32).item().toInt();
                torch::Tensor rot_mat = batch_get_3children_orient_svd(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[in], spine_child);
                rot_mat = rot_mat.to(m__device);
                if (SHOWOUT)
                {
                    std::cout << "rot_mat " << rot_mat << std::endl;
                }

                //rot_mat_chain.append(
                //    torch.matmul(
                //        rot_mat_chain[parents[i]],
                //        rot_mat)
                //)
                auto ind = parents.index({ i }).to(torch::kInt32).item();
                in = ind.toInt();
                try {
                    torch::Tensor ttt = torch::matmul(rot_mat_chain[in].to(torch::kFloat64), rot_mat.to(torch::kFloat64));
                    if (SHOWOUT)
                    {
                        std::cout << "ttt" << ttt << std::endl;
                    }
                }
                catch (std::exception& e)
                {
                    std::cout << e.what() << std::endl;
                    throw;
                }
                rot_mat_chain.push_back(torch::matmul(rot_mat_chain[in].to(torch::kFloat64), rot_mat.to(torch::kFloat64)));
                rot_mat_local.push_back(rot_mat);

            }
            else if (child_indices[i].size() == 1)
            {
                //大部分都执行此条件。
				//only one child
// 				child = child_indices[i][0]
                int child = child_indices[i][0];

                torch::Tensor child_rest_loc = rel_rest_pose.index({ Slice(),child });// [:, child]
                child_rest_loc = child_rest_loc.to(torch::kFloat64);
                if (SHOWOUT)
                {
                    std::cout << "child_rest_loc " << child_rest_loc << std::endl;
                }
                torch::Tensor child_final_loc = rel_pose_skeleton.index({ Slice(), child});// [:, child]
                if (SHOWOUT)
                {
                    std::cout << "child_final_loc " << child_final_loc << std::endl;
                }
                child_final_loc = child_final_loc.to(torch::kFloat64);
                /*
				*child_final_loc = torch.matmul(
				rot_mat_chain[parents[i]].transpose(1, 2),
				child_final_loc)
                */
                auto ind = parents.index({i}).to(torch::kInt32).item(); 
                int in = ind.toInt();
                try
                {
                    child_final_loc = torch::matmul(rot_mat_chain[in].transpose(1, 2),
                        child_final_loc);
                    if (SHOWOUT)
                    {
                        std::cout << "child_final_loc " << child_final_loc << std::endl;
                    }
                }
				catch (std::exception& e)
				{
					std::cout << e.what() << std::endl;
					throw;
				}

                //child_final_norm = torch.norm(child_final_loc, dim = 1, keepdim = True)
                torch::Tensor child_final_norm = torch::norm(child_final_loc, 2, 1, true);
                if (SHOWOUT)
                {
                    std::cout << "child_final_norm " << child_final_norm << std::endl;
                }
                //child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
                torch::Tensor  child_rest_norm = torch::norm(child_rest_loc, 2, 1, true);
                if (SHOWOUT)
                {
                    std::cout << "child_rest_norm " << child_rest_norm << std::endl;
                }


                //axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
                torch::Tensor axis = torch::cross(child_rest_loc, child_final_loc, 1);
                if (SHOWOUT)
                {
                    std::cout << "axis" << axis << std::endl;
                }

                //axis_norm = torch.norm(axis, dim=1, keepdim=True)
                torch::Tensor axis_norm = torch::norm(axis,2, 1,true);
                if (SHOWOUT)
                {
                    std::cout << "axis_norm" << axis_norm << std::endl;
                }

                //temp = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True)
                torch::Tensor temp = torch::sum(child_rest_loc * child_final_loc, 1, true);
                if (SHOWOUT)
                {
                    std::cout << "temp" << temp << std::endl;
                }


                torch::Tensor angle = torch::acos(temp / (child_rest_norm * child_final_norm));
                if(SHOWOUT)
                { 
                    std::cout << "angle" << angle << std::endl;
                }

				//axis = axis / (axis_norm + 1e-8)
				//aa = angle * axis
                axis = axis / (axis_norm + 1e-8);
                torch::Tensor aa = angle * axis;
				if (SHOWOUT)
				{
					std::cout << "aa" << aa << std::endl;
				}
                torch::Tensor tt = torch::squeeze(aa);
				if (SHOWOUT)
				{
					std::cout << "tt: " << tt << std::endl;
				}
                auto x = tt.index({ 0 }).to(torch::kFloat64).item();
                float xx = x.toFloat();
				auto y = tt.index({ 1 }).to(torch::kFloat64).item();
				float yy = y.toFloat();
				auto z = tt.index({ 2 }).to(torch::kFloat64).item();
				float zz = z.toFloat();

				Mat src = (Mat_<double>(3, 1) << xx, yy, zz);
                if (SHOWOUT)
                {
                    std::cout << src << std::endl;
                }
                //rot_mat = cv2.Rodrigues(aa.detach().cpu().numpy())[0]
                
				//Mat auxRinv = Mat::eye(3, 3, CV_32F);
				//Rodrigues(Rc1, auxRinv);
                //rot_mat = cv.Rodrigues(aa.detach().cpu().numpy())[0]
                cv::Mat dst;
                cv::Rodrigues(src, dst);
                if (SHOWOUT)
                {
                    std::cout << dst << std::endl;
                }
                torch::Tensor rot_mat = torch::from_blob(dst.data, { 3, 3 }, torch::kFloat64);
                if (SHOWOUT)
                {
                    std::cout << "rot_mat: " << rot_mat.sizes() << rot_mat << std::endl;
                }

                rot_mat = rot_mat.to(m__device);
                rot_mat = torch::unsqueeze(rot_mat,0);
                if (SHOWOUT)
                {
                    std::cout << "rot_mat: " << rot_mat.sizes() << rot_mat << std::endl;

                }
                /*
				*rot_mat_chain.append(
				torch.matmul(
					rot_mat_chain[parents[i]],
					rot_mat)
			)
			rot_mat_local.append(rot_mat)

                */
                in = parents.index({ i }).to(torch::kInt32).item().toInt();
                torch::Tensor tempp = rot_mat_chain[in];
                if (SHOWOUT)
                {
                    std::cout << "rot_mat_chain[in]" << rot_mat_chain[in] << std::endl;
                }
                torch::Tensor result;
                try
                {
                    /*torch::Tensor*/ result = torch::matmul(rot_mat_chain[in].to(torch::kFloat64), rot_mat);
					if (SHOWOUT)
					{
						//torch::Tensor tempp = rot_mat_chain[in];
						std::cout << "result" << result << std::endl;
					}
                }
                catch (std::exception& e)
                {
                    std::cout << e.what() << std::endl;
                    throw;
                }

				
                
                rot_mat_chain.push_back(result);
                rot_mat_local.push_back(rot_mat);                

            }
        }//end for 1

        //rot_mats = torch.stack(rot_mat_local, dim = 1)
        torch::Tensor rot_mats = torch::stack(rot_mat_local, 1);
        if (SHOWOUT)
        {
            std::cout << "rot_mats" << rot_mats << std::endl;
        }
        return rot_mats;//temperary
        //std::cout << "rel_rest_pose" << rel_rest_pose << std::endl;

    }

    torch::Tensor LinearBlendSkinning::batch_get_pelvis_orient_svd(
        const torch::Tensor& rel_pose_skeleton,
        const torch::Tensor& rel_rest_pose,
        const torch::Tensor& parents,
        const torch::Tensor& children
        //dtype
    )
    {
        //pelvis_child = [int(children[0])]
        //std::cout << children.sizes() << children << children.index({ 0 }) << children.index({ 0 }).item()<<std::endl;
        //auto pelvis_child = children.index({ 0 }).to(torch::kInt32).item();// .item();
        //std::cout << children.sizes() << children << children.dtype() << std::endl;
        //int pelvis_child = int(children.index({Slice(0)}).to(torch::kI32));
        std::vector<int> pelvis_child;
        //std::vector<int> child
        pelvis_child.push_back(1);
        for (int i = 1; i < 29; i++)
        {
            int nCount = std::count(pelvis_child.begin(), pelvis_child.end(), i);
            if (parents.index({ i }).item() == 0 && nCount <= 0)
            {
                pelvis_child.push_back(i);
            }
        }
        /*
		rest_mat = []
	    target_mat = []
	    for child in pelvis_child:
		rest_mat.append(rel_rest_pose[:, child].clone())
		target_mat.append(rel_pose_skeleton[:, child].clone())
        */
        torch::Tensor temp = torch::squeeze(rel_rest_pose);
        if (SHOWOUT)
        {
            std::cout << "rel_rest_pose temp all:" << temp << std::endl;
        }

        torch::Tensor rest_mat = torch::zeros({9});
        torch::Tensor target_mat = torch::zeros({ 9 });
        int i = 0;
        for (std::vector<int>::iterator itr =pelvis_child.begin(); itr != pelvis_child.end(); itr++ )
        {
            
            int child = *itr;
            //rest_mat.index({ Slice(i * 3, i * 3 + 3) }).print();
            torch::Tensor temp = rel_rest_pose.index({ Slice(),child }).clone();//.print();
            temp = torch::squeeze(temp);
            if (SHOWOUT)
            {
                std::cout << "temp:" << temp << std::endl;
            }

			torch::Tensor temp2 = rel_pose_skeleton.index({ Slice(),child }).clone();
			temp2 = torch::squeeze(temp2);
            if (SHOWOUT)
            {
                std::cout << "temp2:" << temp2 << std::endl;
            }

            //std::cout << "rel_rest_pose:" << rel_rest_pose << std::endl;
            rest_mat.index({ Slice(i * 3, i * 3 + 3) }) = temp;// rel_rest_pose.index({ Slice(),child }).clone();// [:, child] .clone()
            target_mat.index({ Slice(i * 3, i * 3 + 3) }) = temp2;// rel_rest_pose.index({ Slice(),child }).clone();// [:, child] .clone()

            //rest_mat.print();
            if (SHOWOUT)
            {
                std::cout << "rest_mat:" << rest_mat << std::endl;
                std::cout << "target_mat:" << target_mat << std::endl;
            }

            //rel_rest_pose.index({ Slice(),child }).print();
            i++;
        }
        
        rest_mat = torch::reshape(rest_mat, { 1,3,3 });// .transpose(0, 1);
        rest_mat = torch::squeeze(rest_mat).transpose(0,1);
        rest_mat = torch::unsqueeze(rest_mat, 0);
        if (SHOWOUT)
        {
            std::cout << "rest_mat:" << rest_mat.sizes() << rest_mat << std::endl;
        }
        target_mat = torch::reshape(target_mat, { 1,3,3 });
        target_mat = torch::squeeze(target_mat).transpose(0, 1);
        target_mat = torch::unsqueeze(target_mat, 0);
        if (SHOWOUT)
        {
            std::cout << "target_mat:" << target_mat.sizes() << target_mat << std::endl;
        }


        //S = rest_mat.bmm(target_mat.transpose(1, 2))
        torch::Tensor S = rest_mat.bmm(target_mat.transpose(1, 2));
        if (SHOWOUT)
        {
            std::cout << "S:" << S.sizes() << S << std::endl;
        }

        
//         mask_zero = S.sum(dim=(1, 2))
//         torch::Tensor mask_zero = S.sum((1, 2),false);        
//         std::cout << mask_zero.sizes() << mask_zero << std::endl;
// 
//         torch::Tensor mask_zero2 = torch::sum(S, (1, 2),true);
//         std::cout <<"mask_zero2:" << mask_zero2.sizes() << mask_zero2 << std::endl;

		//device = S_non_zero.device
        torch::Tensor U, Sigma, V;
        std::tuple t1;
        std::tuple<at::Tensor, at::Tensor, at::Tensor> t2;
		//U, Sigma,

        t2 = torch::svd(S.cpu());

        U = std::get<0>(t2);
        V = std::get<2>(t2);
        if (SHOWOUT)
        {
            std::cout << "V:" << V.sizes() << V.device() << V << std::endl;
        }

        U = U.to(m__device);
        V = V.to(m__device);
        if (SHOWOUT)
        {
            std::cout << "V:" << V.sizes() << V.device() << V << std::endl;
        }
//		U = U.to(device = device)
//		V = V.to(device = device)



        torch::Tensor rot_mat = torch::zeros_like(S);
        torch::Tensor det_u_v = torch::det(torch::bmm(V, U.transpose(1, 2)));
        if (SHOWOUT)
        {
            std::cout << "det_u_v:" << det_u_v.sizes() << det_u_v.device() << det_u_v << std::endl;
        }

        torch::Tensor det_modify_mat = torch::eye(3, U.device()).unsqueeze(0);// .expand(U.size(0), -1, -1).clone()
        if (SHOWOUT)
        {
            std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;
        }
        det_modify_mat.index_put_({ Slice(),2,2 }, det_u_v);// = det_u_v;
        if (SHOWOUT)
        {
            std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;
        }

        //rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))
        torch::Tensor rot_mat_non_zero = torch::bmm(torch::bmm(V, det_modify_mat), U.transpose(1, 2));
        if (SHOWOUT)
        {
            std::cout << "rot_mat_non_zero:" << rot_mat_non_zero.sizes() << rot_mat_non_zero.device() << rot_mat_non_zero << std::endl;
        }
        return rot_mat_non_zero;
    }

    torch::Tensor LinearBlendSkinning::quaternion_to_angle_axis(torch::Tensor& quaternion)
    {
		//# unpack input and compute conversion
        torch::Tensor q1 = quaternion.index({ "...",1 });// .index("...", 1);// [..., 1] ;
        if (SHOWOUT)
        {
            std::cout << "q1 " << q1 << std::endl;
        }

        torch::Tensor q2 = quaternion.index({ "...", 2 });// [..., 2]
        if (SHOWOUT)
        {
            std::cout << "q2 " << q2 << std::endl;
        }

        //q3 : torch.Tensor = quaternion[..., 3]
        torch::Tensor q3 = quaternion.index({ "...",3 });// [..., 3] ;
        if (SHOWOUT)
        {
            std::cout << "q3 " << q3 << std::endl;
        }

        torch::Tensor sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
        if (SHOWOUT)
        {
            std::cout << "sin_squared_theta " << sin_squared_theta << std::endl;
        }

        //sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
        torch::Tensor sin_theta = torch::sqrt(sin_squared_theta);
        if (SHOWOUT)
        {
            std::cout << "sin_theta" << sin_theta << std::endl;
        }

         //cos_theta: torch.Tensor = quaternion[..., 0]
        torch::Tensor cos_theta = quaternion.index({ "...",0 });// [..., 0]
        if (SHOWOUT)
        {
            std::cout << "cos_theta" << cos_theta << std::endl;
        }

// 	two_theta: torch.Tensor = 2.0 * torch.where(
// 		cos_theta < 0.0,
// 		torch.atan2(-sin_theta, -cos_theta),
// 		torch.atan2(sin_theta, cos_theta))
        torch::Tensor two_theta = 2.0 * torch::where(
            cos_theta < 0.0,
            torch::atan2(-sin_theta, -cos_theta),
            torch::atan2(sin_theta, cos_theta));
        if (SHOWOUT)
        {
            std::cout << "two_theta" << two_theta << std::endl;
        }
        

        torch::Tensor k_pos = two_theta / sin_theta;
        torch::Tensor k_neg = 2.0 * torch::ones_like(sin_theta);
        torch::Tensor k = torch::where(sin_squared_theta > 0.0, k_pos, k_neg);
        if (SHOWOUT)
        {
            std::cout << "k" << k << std::endl;
        }

        //angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
        torch::Tensor angle_axis = torch::zeros_like(quaternion).index({ "...", Slice(None,3) });// [..., :3]
        if (SHOWOUT)
        {
            std::cout << "angle_axis" << angle_axis << std::endl;
        }

		//angle_axis[..., 0] += q1 * k
        angle_axis.index({ "...", 0 }) += q1 * k;
        angle_axis.index({ "...", 1 }) += q2 * k;
        angle_axis.index({ "...",2 }) += q3 * k;
        if (SHOWOUT)
        {
            std::cout << "angle_axis" << angle_axis << std::endl;
        }
        return angle_axis;


        

    }
    torch::Tensor LinearBlendSkinning::rotation_matrix_to_quaternion(torch::Tensor& rotation_matrix,double eps)
    {
        torch::Tensor rmat_t = torch::transpose(rotation_matrix, 1, 2);
        if (SHOWOUT)
        {
            std::cout << "rmat_t:" << rmat_t.sizes() << rmat_t.device() << std::endl;
        }

        torch::Tensor mask_d2 = rmat_t.index({ Slice(), 2, 2 }) < eps;
        if (SHOWOUT)
        {
            std::cout << "mask_d2 " << mask_d2.sizes() << mask_d2 << std::endl;
        }

        torch::Tensor  mask_d0_d1 = rmat_t.index({ Slice(), 0, 0 }) > rmat_t.index({ Slice(),1,1 });
        if (SHOWOUT)
        {
            std::cout << "mask_d0_d1" << mask_d0_d1 << std::endl;
        }

        torch::Tensor  mask_d0_nd1 = rmat_t.index({ Slice(), 0, 0 }) < -rmat_t.index({ Slice(),1,1 });
        if (SHOWOUT)
        {
            std::cout << "mask_d0_nd1" << mask_d0_nd1 << std::endl;
        }
    
        torch::Tensor t0 = 1 + rmat_t.index({ Slice(), 0, 0 }) - rmat_t.index({ Slice(), 1, 1 }) - rmat_t.index({ Slice(), 2, 2});
        if (SHOWOUT)
        {
            std::cout << "t0" << t0 << std::endl;
        }
        
        torch::Tensor tem1 = rmat_t.index({ Slice(), 1,2 }) - rmat_t.index({ Slice(),2,1 });// [:, 2, 1]
        torch::Tensor tem2 = rmat_t.index({ Slice(), 0, 1 }) + rmat_t.index({ Slice(), 1,0 });//[:, 1, 0]
        torch::Tensor tem3 = rmat_t.index({ Slice(), 2, 0 }) + rmat_t.index({ Slice(),0,2 });// [:, 0, 2]


        torch::Tensor q0 = torch::stack({ tem1, t0, tem2,tem3 }, -1);
        if (SHOWOUT)
        {
            std::cout << "q0" << q0 << std::endl;
        }

        torch::Tensor  t0_rep = t0.repeat({ 4, 1 }).t();
        if (SHOWOUT)
        {
            std::cout << "t0_rep" << t0_rep << std::endl;
        }

        torch::Tensor t1 = 1 - rmat_t.index({ Slice(),0,0 })/*[:, 0, 0] */ + rmat_t.index({ Slice(),1,1 })/*[:, 1, 1] */ - rmat_t.index({Slice(),2,2});// [:, 2, 2]
        if (SHOWOUT)
        {
            std::cout << "t1" << t1 << std::endl;
        }

        torch::Tensor ttt0 = rmat_t.index({ Slice(),2,0 })/*[:, 2, 0] */ - rmat_t.index({ Slice(), 0, 2 });// [:, 0, 2] ;
        torch::Tensor ttt1 = rmat_t.index({ Slice(), 0, 1 })/*[:, 0, 1] */ + rmat_t.index({ Slice(),1,0 });// [:, 1, 0]
        torch::Tensor ttt2 = rmat_t.index({ Slice(),1,2 })/*[:, 1, 2]*/ + rmat_t.index({ Slice(),2,1 });// [:, 2, 1]
        torch::Tensor q1 = torch::stack({ ttt0, ttt1,	t1, ttt2 }, -1);
        if (SHOWOUT)
        {
            std::cout << "q1" << q1 << std::endl;
        }

        torch::Tensor t1_rep = t1.repeat({ 4, 1 }).t();
        if (SHOWOUT)
        {
            std::cout << "t1_rep" << t1_rep << std::endl;
        }

        torch::Tensor t2 = 1 - rmat_t.index({ Slice(),0,0 })/*[:, 0, 0]*/ - rmat_t.index({ Slice(),1,1 })/*[:, 1, 1]*/ + rmat_t.index({ Slice(),2,2 });// [:, 2, 2]
        if (SHOWOUT)
        {
            std::cout << "t2" << t2 << std::endl;
        }

        torch::Tensor sss0 = rmat_t.index({ Slice(),0,1 })/*[:, 0, 1] */ - rmat_t.index({ Slice(),1,0 });// [:, 1, 0] ;
        torch::Tensor sss1 = rmat_t.index({ Slice(),2,0 })/*[:, 2, 0]*/ + rmat_t.index({ Slice(), 0,2 });// [:, 0, 2] ;
        torch::Tensor sss2 = rmat_t.index({ Slice(), 1,2 })/*[:, 1, 2]*/ + rmat_t.index({ Slice(),2,1 });// [:, 2, 1] ;

        torch::Tensor q2 = torch::stack({ sss0, sss1,sss2,t2 }, -1);
        if (SHOWOUT)
        {
            std::cout << "q2" << q2 << std::endl;
        }

        torch::Tensor t2_rep = t2.repeat({ 4, 1 }).t();
        if (SHOWOUT)
        {
            std::cout << "t2_rep" << t2_rep << std::endl;
        }

        torch::Tensor t3 = 1 + rmat_t.index({ Slice(),0,0 })/*[:, 0, 0]*/ + rmat_t.index({ Slice(),1,1 })/*[:, 1, 1]*/ + rmat_t.index({ Slice(),2,2 });// [:, 2, 2]
        if (SHOWOUT)
        {
            std::cout << "t3" << t3 << std::endl;
        }

        torch::Tensor yyy0 = rmat_t.index({ Slice(), 1,2 })/*[:, 1, 2]*/ - rmat_t.index({ Slice(),2,1 });// [:, 2, 1] ;
        torch::Tensor yyy1 = rmat_t.index({ Slice(),2,0 })/*[:, 2, 0]*/ - rmat_t.index({ Slice(),0,2 });// [:, 0, 2] ;
        torch::Tensor yyy2 = rmat_t.index({ Slice(),0,1 })/*[:, 0, 1]*/ - rmat_t.index({ Slice(),1,0 });// [:, 1, 0] ;

        torch::Tensor q3 = torch::stack({ t3, yyy0, yyy1,yyy2 }, -1);
        if (SHOWOUT)
        {
            std::cout << "q3" << q3 << std::endl;
        }

        torch::Tensor t3_rep = t3.repeat({ 4, 1 }).t();
        if (SHOWOUT)
        {
            std::cout << "t3_rep" << t3_rep << std::endl;
        }

        torch::Tensor mask_c0 = mask_d2 * mask_d0_d1;
        if (SHOWOUT)
        {
            std::cout << "mask_c0 " << mask_c0 << std::endl;
        }
        torch::Tensor mask_c1 = mask_d2 * ~mask_d0_d1;
        if (SHOWOUT)
        {
            std::cout << "mask_c1 " << mask_c1 << std::endl;
        }

        torch::Tensor mask_c2 = ~mask_d2 * mask_d0_nd1;
        if (SHOWOUT)
        {
            std::cout << "mask_c2 " << mask_c2 << std::endl;
        }

        torch::Tensor mask_c3 = ~mask_d2 * ~mask_d0_nd1;
        if (SHOWOUT)
        {
            std::cout << "mask_c3 " << mask_c3 << std::endl;
        }

        mask_c0 = mask_c0.view({ -1, 1 }).type_as(q0);
        if (SHOWOUT)
        {
            std::cout << "mask_c0 " << mask_c0 << std::endl;
        }
        mask_c1 = mask_c1.view({ -1, 1 }).type_as(q1);
        if (SHOWOUT)
        {
            std::cout << "mask_c1 " << mask_c1 << std::endl;
        }
        mask_c2 = mask_c2.view({ -1, 1 }).type_as(q2);
        if (SHOWOUT)
        {
            std::cout << "mask_c2 " << mask_c2 << std::endl;
        }
        mask_c3 = mask_c3.view({ -1, 1 }).type_as(q3);
        if (SHOWOUT)
        {
            std::cout << "mask_c3 " << mask_c3 << std::endl;
        }


        torch::Tensor q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3;
        if (SHOWOUT)
        {
            std::cout << "q " << q << std::endl;
        }
        q /= torch::sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +
            t2_rep * mask_c2 + t3_rep * mask_c3);
        if (SHOWOUT)
        {
            std::cout << "q " << q << std::endl;
        }
        q *= 0.5;
        
        return q;

    }


    torch::Tensor LinearBlendSkinning::vertices2joints(const torch::Tensor& J_regressor, const torch::Tensor& vertices)
    {
        return torch::einsum("bik, ji->bjk", { vertices, J_regressor });
    }
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
// 
    torch::Tensor LinearBlendSkinning::batch_get_3children_orient_svd(
        std::vector<torch::Tensor> rel_pose_skeleton,
        std::vector<torch::Tensor> rel_rest_pose,
        torch::Tensor rot_mat_chain_parent,
        std::vector<int> children_list)
    {
// 		rest_mat = []
// 		target_mat = []
        std::vector<torch::Tensor> rest_mat;
        std::vector<torch::Tensor> target_mat;

// 		for c, child in enumerate(children_list) :
// 			if isinstance(rel_pose_skeleton, list) :
// 				target = rel_pose_skeleton[c].clone()
// 				template = rel_rest_pose[c].clone()
// 			else :
// 				target = rel_pose_skeleton[:, child].clone()
// 				template = rel_rest_pose[:, child].clone()
// 
// 				target = torch.matmul(
// 					rot_mat_chain_parent.transpose(1, 2),
// 					target)
// 
// 				target_mat.append(target)
// 				rest_mat.append(template)

        int c = 0;
        for (std::vector<int>::iterator iter = children_list.begin(); iter != children_list.end(); iter++)
        {
            int child = *iter;
// 			target = rel_pose_skeleton[c].clone()
// 				template = rel_rest_pose[c].clone()
            torch::Tensor target = rel_pose_skeleton[c].clone();
            torch::Tensor templates = rel_rest_pose[c].clone();
            if (SHOWOUT)
            {
                std::cout << "templates" << templates << std::endl;
                std::cout << "rot_mat_chain_parent" << rot_mat_chain_parent << std::endl;
                std::cout << "rot_mat_chain_parent.transpose(1, 2)" << rot_mat_chain_parent.transpose(1, 2) << std::endl;
                std::cout << "target" << target << std::endl;
            }
            torch::Tensor target_new;
            target_new = torch::matmul(
                rot_mat_chain_parent.transpose(1, 2), target);
            if (SHOWOUT)
            {
                std::cout << "target_new" << target_new << std::endl;
            }
            target_mat.push_back(target_new);
            rest_mat.push_back(templates);
            c++;
        }

// 		rest_mat = torch.cat(rest_mat, dim = 2)
        //torch::Tensor rest_mat_new = torch::cat(rest_mat,2);
        if (SHOWOUT)
        {
            std::cout << "rest_mat" << rest_mat << std::endl;
        }

        torch::Tensor rest_mat_new2 = torch::stack(rest_mat, 1);        
        if (SHOWOUT)
        {
            std::cout << "rest_mat_new2" << rest_mat_new2 << std::endl;
        }
        torch::Tensor rest_mat_temp = torch::squeeze(rest_mat_new2);// .transpose();
        rest_mat_temp = rest_mat_temp.transpose(0, 1);
        rest_mat_temp = torch::unsqueeze(rest_mat_temp, 0);
        if (SHOWOUT)
        {
            std::cout << "rest_mat_temp" << rest_mat_temp << std::endl;
        }
        if (SHOWOUT)
        {
            std::cout << "target_mat" << target_mat << std::endl;
        }
        torch::Tensor target_mat_new2 = torch::stack(target_mat, 1);
        if (SHOWOUT)
        {
            std::cout << "target_mat_new2" << target_mat_new2 << std::endl;
        }
        torch::Tensor target_mat_temp = torch::squeeze(target_mat_new2);
        target_mat_temp = target_mat_temp.transpose(0, 1);
        if (SHOWOUT)
        {
            std::cout << "target_mat_temp" << target_mat_temp << std::endl;
        }
        target_mat_temp = torch::unsqueeze(target_mat_temp,0);
        if (SHOWOUT)
        {
            std::cout << "target_mat_temp" << target_mat_temp << std::endl;
        }

        
        torch::Tensor S = rest_mat_temp.bmm(target_mat_temp.transpose(1, 2));
        if (SHOWOUT)
        {
            std::cout << "S" << S << std::endl;
        }
        /*
		*     device = S.device
	        U, _, V = torch.svd(S.cpu())
	        U = U.to(device=device)
	        V = V.to(device=device)
        */
        torch::Tensor U, Sigma, V;
        std::tuple<at::Tensor, at::Tensor, at::Tensor> t2;
        t2 = torch::svd(S.cpu());

        U = std::get<0>(t2);
        V = std::get<2>(t2);
        if (SHOWOUT)
        {
            std::cout << "V:" << V.sizes() << V.device() << V << std::endl;
        }


        /*
        * 
		*     
	
            det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

        */
        torch::Tensor det_u_v = torch::det(torch::bmm(V, U.transpose(1, 2)));
        if (SHOWOUT)
        {
            std::cout << "det_u_v:" << det_u_v.sizes() << det_u_v.device() << det_u_v << std::endl;
        }
        torch::Tensor det_modify_mat = torch::eye(3, U.device()).unsqueeze(0);// .expand(U.size(0), -1, -1).clone()
        if (SHOWOUT)
        {
            std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;
        }
        det_modify_mat.index_put_({ Slice(),2,2 }, det_u_v);// = det_u_v;
        if (SHOWOUT)
        {
            std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;
        }
        //rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))
        //rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))
        torch::Tensor first = torch::bmm(V, det_modify_mat.to(torch::kFloat64));
        if (SHOWOUT)
        {
            std::cout << "first:" << first.sizes() << first.device() << first << std::endl;
        }
        torch::Tensor rot_mat = torch::bmm(torch::bmm(V, det_modify_mat.to(torch::kFloat64)), U.transpose(1, 2).to(torch::kFloat64));
        if (SHOWOUT)
        {
            std::cout << "rot_mat:" << rot_mat.sizes() << rot_mat.device() << rot_mat << std::endl;
        }

        
        return rot_mat; //临时

    }



    
    void LinearBlendSkinning::write_json(ofstream &myfile, const int id, const torch::Tensor& Rh, const torch::Tensor& Th, const torch::Tensor& poses, const torch::Tensor& shapes)
    {
        /*
		* quat = quat.to(torch::kCPU);
		ofstream  myfile("data/000000.json");
		//out_text.append('[\n')
		myfile << "[\n";

		double* ptr = (double*)quat.data_ptr();
		for (size_t i = 0; i < 72; i++) {
			try
			{
				//std::cout << *((ptr + i)) << std::endl;
				myfile << *((ptr + i));
				myfile << ", ";

			}
			catch (const std::exception& e)
			{
				std::cout << e.what() << std::endl;
				throw;
			}

		}
		myfile << "]\n";
		myfile.close();

        */
        myfile << "{\n";
        myfile << "\"id\":" << id << ",\n";
        myfile << "\"Rh\": " << "[\n";
        myfile << "[";
        if (SHOWOUT)
        {
            std::cout << "Rh" << Rh << std::endl;
            std::cout << "Th" << Th << std::endl;
            std::cout << "Th" << poses << std::endl;
            std::cout << "Th" << shapes << std::endl;
            
        }
        float* ptr = (float*)Rh.data_ptr();
        for (size_t i = 0; i < 2; i++)
        {
			myfile << (float)*((ptr + i));
			myfile << ", ";
        }
        myfile << *((ptr + 2));
        myfile << "]\n],\n";

        myfile << "\"Th\": " << "[\n";
        myfile << "[";
		ptr = (float*)Th.data_ptr();
		for (size_t i = 0; i < 2; i++)
		{
			myfile << (float)*((ptr + i));
			myfile << ", ";
		}
		myfile << (float)*((ptr + 2));
		myfile << "]\n],\n";

        myfile << "\"poses\": " << "[\n";
        myfile << "[";
        double* ptr2 = (double*)poses.data_ptr();
        for (size_t i = 0; i < 71; i++)
        {
            myfile << *((ptr2 + i));
            myfile << ", ";
        }
        myfile << *((ptr2 + 71));
        myfile << "]\n],\n";

		myfile << "\"shapes\": " << "[\n";
		myfile << "[";
		ptr = (float*)shapes.data_ptr();
		for (size_t i = 0; i < 9; i++)
		{
			myfile << *((ptr + i));
			myfile << ", ";
		}
		myfile << *((ptr + 9));
		myfile << "]\n]\n";
        myfile << "}\n";            


    }


/*
	def umeyama(src, dst, estimate_scale) 
		References
		----------
		..[1] "Least-squares estimation of transformation parameters between two
		point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
		"""

		num = src.shape[0]
		dim = src.shape[1]

		# Compute mean of src and dst.
		src_mean = src.mean(axis = 0)
		dst_mean = dst.mean(axis = 0)

		# Subtract mean from src and dst.
		src_demean = src - src_mean
		dst_demean = dst - dst_mean

		# Eq. (38).
		A = np.dot(dst_demean.T, src_demean) / num

		# Eq. (39).
		d = np.ones((dim, ), dtype = np.double)
		if np.linalg.det(A) < 0:
	d[dim - 1] = -1

		T = np.eye(dim + 1, dtype = np.double)

		U, S, V = np.linalg.svd(A)

		# Eq. (40) and (43).
		rank = np.linalg.matrix_rank(A)
		if rank == 0:
	return np.nan * T
		elif rank == dim - 1 :
		if np.linalg.det(U)* np.linalg.det(V) > 0:
	T[:dim, : dim] = np.dot(U, V)
		else :
			s = d[dim - 1]
			d[dim - 1] = -1
			T[:dim, : dim] = np.dot(U, np.dot(np.diag(d), V))
			d[dim - 1] = s
		else:
	T[:dim, : dim] = np.dot(U, np.dot(np.diag(d), V.T))

		if estimate_scale :
			# Eq. (41) and (42).
			scale = 1.0 / src_demean.var(axis = 0).sum() * np.dot(S, d)
		else:
	scale = 1.0

		# the python umeyama get a wrong rotation in some unknow condition
		# so we compare the two results and choose the minimum
		# it will be fixed in the future
		homo_src = np.insert(src, 3, 1, axis = 1).T
		rot = T[:dim, : dim]

		rots = []
		losses = []
		for i in range(2) :
			if i == 1 :
				rot[:, : 2] *= -1
				transform = np.eye(dim + 1, dtype = np.double)
				transform[:dim, : dim] = rot * scale
				transform[:dim, dim] = dst_mean - scale * np.dot(T[:dim, : dim], src_mean.T)
				transed = np.dot(transform, homo_src).T[:, : 3]
				loss = np.linalg.norm(transed - dst)
				losses.append(loss)
				rots.append(rot.copy())
				# T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, : dim], src_mean.T)
				# T[:dim, : dim] *= scale
				# print(T)

				# since only the smpl parameters is needed, we return rotation, translation and scale
				trans = dst_mean - scale * np.dot(T[:dim, : dim], src_mean.T)
				if losses[0] > losses[1]:
	rot = rots[1]
				else:
	rot = rots[0]

		return rot, trans, scale
        */
    std::tuple<torch::Tensor, torch::Tensor> LinearBlendSkinning::umeyama(torch::Tensor src, torch::Tensor dst)
    {
// 		"""Estimate N-D similarity transformation with or without scaling.
// 			Parameters
// 			----------
// 			src : (M, N) array
// 			Source coordinates.
// 			dst : (M, N) array
// 			Destination coordinates.
// 			estimate_scale : bool
// 			Whether to estimate scaling factor.
// 			Returns
// 			------ -
// 			T : (N + 1, N + 1)
// 			The homogeneous similarity transformation matrix.The matrix contains
// 			NaN values only if the problem is not well - conditioned.

        int num = src.size(0);// [0] ;
        int dim = src.size(1);
		// Compute mean of src and dst.
        torch::Tensor src_mean = src.mean(0);
        if (SHOWOUT)
        {
            std::cout<<"src_mean" << src_mean << std::endl;
        }
        torch::Tensor dst_mean = dst.mean(0);
		if (SHOWOUT)
		{
			std::cout <<"src_mean" << dst_mean << std::endl;
		}

		//# Subtract mean from src and dst.
        torch::Tensor 	src_demean = src - src_mean;
		if (SHOWOUT)
		{
			std::cout << "src_demean " << src_demean << std::endl;
		}
        torch::Tensor 	dst_demean = dst - dst_mean;
		if (SHOWOUT)
		{
			std::cout << "dst_demean  " << dst_demean << std::endl;
		}
     

        torch::Tensor 	A = torch::mm(torch::t(dst_demean), src_demean) / num;
		if (SHOWOUT)
		{
			std::cout << "A" << A << std::endl;
		}
        torch::Tensor d = torch::ones({ dim });// , dtype = np.double)

        //torch::linalg.det(A);
		if (torch::linalg::det(A).item().toFloat() < 0)
        {
            d[dim - 1] = -1;
        }
		if (SHOWOUT)
		{
			std::cout <<"d" << torch::linalg::det(A).item().toFloat() << "d " << d << std::endl;
		}

        torch::Tensor 	T = torch::eye(dim + 1);// , dtype = np.double)

		if (SHOWOUT)
		{
			std::cout << "T" << T << std::endl;
		}
        std::tuple<at::Tensor, at::Tensor, at::Tensor> t;
        t = torch::svd(A);

        torch::Tensor U = std::get<0>(t);
        torch::Tensor S = std::get<1>(t);
        torch::Tensor V = std::get<2>(t);
        V = V.t();
		if (SHOWOUT)
		{
			std::cout << "U" << U << std::endl;
            std::cout << "S" << S << std::endl;
            std::cout << "V" << V << std::endl;
		}


        int rank = torch::linalg::matrix_rank(A, 0 ,true).item().toInt();//anzs

        if (rank == dim - 1)
        {
            if ((torch::linalg::det(U) * torch::linalg::det(V)).item().toInt() > 0)
            {
                T.index({Slice(None,dim), Slice(None,dim)}) = torch::dot(U, V);
            }       
            else
            {
                torch::Tensor s = d[dim - 1];
                d[dim - 1] = -1;
                T.index({Slice(None, dim), Slice(None, dim)}) = torch::dot(U, torch::dot(torch::diag(d), V));
                d[dim - 1] = s;
            }
         }
        else
        {
            T.index({Slice(None,dim), Slice(None, dim)}) = torch::mm(U, torch::mm(torch::diag(d), V.transpose(0,1)));
			if (SHOWOUT)
			{
				std::cout << "T" << T << std::endl;
			}

         }

        float scale = 1.0f;

        /* Is there a way to insert a tensor into an existing tensor? 
        * https://discuss.pytorch.org/t/is-there-a-way-to-insert-a-tensor-into-an-existing-tensor/14642
		* a = torch.rand(3, 4)
        b = torch.zeros(1, 4)

        idx = 1
        c = torch.cat([a[:idx], b, a[idx:]], 0)
        */

        torch::Tensor homo_src;
        //torch::Tensor homo_src = torch::cat(src, 3, 1, 1).T; //在第三列上插入1
        torch::Tensor src_t = src.t();// index({ Slice(None,3) });
        torch::Tensor last_one_row = torch::ones({1, src_t.size(1) });
        homo_src = torch::cat({ src_t,last_one_row }, 0);
		if (SHOWOUT)
		{
			std::cout << "src" << src << std::endl;
            std::cout << "src_t" << src_t << std::endl;
            std::cout << "last_one_row" << last_one_row << std::endl;
            std::cout << "homo_src" << homo_src << std::endl;
		}
    
        torch::Tensor  rot = T.index({Slice(None,dim),Slice(None,dim)});// [:dim, : dim] ;
        if (SHOWOUT)
        {
            std::cout << "rot" << rot << std::endl;
        }

        std::vector< torch::Tensor> rots;// = [];
        std::vector< float>	losses;// = []
        for (int i = 0; i < 2; i++)
        {
            if (i == 1)
            {
                rot.index({Slice(), Slice(None,2)})/*[:, : 2] */ *= -1;
                rot.index({ Slice(), Slice(None,2) }) *= -1;
				if (SHOWOUT)
				{
					std::cout << "rot" << rot << std::endl;
				}
            }
            //transform = np.eye(dim + 1, dtype = np.double)
            torch::Tensor transform = torch::eye(dim + 1);
            transform.index({ Slice(None,dim),Slice(None,dim) }) = rot * scale;                        
			if (SHOWOUT)
			{
				std::cout << "transform" << transform << std::endl;

                std::cout << "src_mean.t()" << src_mean.t().reshape({ -1, 1 }) << std::endl;

                torch::Tensor ttt = T.index({ Slice(None,dim), Slice(None,dim) });
                std::cout << "ttt" << ttt << std::endl;
                
			}

            
			if (SHOWOUT)
			{
				torch::Tensor tttt = torch::mm(T.index({ Slice(None,dim), Slice(None,dim) }), src_mean.t().reshape({ -1, 1 }));
				//torch::Tensor ttt = T.index({ Slice(None,dim), Slice(None,dim) });
				std::cout << "ttt" << tttt << std::endl;
			}

            //transform[:dim,dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
            try
            {               
                transform.index({ Slice(None, dim),dim }) = dst_mean - scale * torch::mm(T.index({ Slice(None,dim), Slice(None,dim) }), src_mean.t().reshape({ -1, 1 })).squeeze();
            }
            catch (const std::exception& e)
            {
                std::cout << e.what() << std::endl;
                throw;
            }
            if (SHOWOUT)
            {
                std::cout << "transform" << transform << std::endl;
            }
            

            torch::Tensor transed = torch::mm(transform, homo_src).t().index({Slice(), Slice(None, 3)});// [:, : 3]
            float loss = torch::norm(transed - dst, 2).item().toFloat();
            //losses.append(loss);
            losses.push_back(loss);
            //rots.append(rot.copy())
            rots.push_back(rot);

        }

		//# since only the smpl parameters is needed, we return rotation, translation and scale
	//# since only the smpl parameters is needed, we return rotation, translation and scale
        torch::Tensor  trans;
        try {
              trans = dst_mean - scale * torch::mm(T.index({ Slice(None, dim),Slice(None,dim) }), src_mean.t().reshape({ -1, 1 })).squeeze();// [:dim, : dim] , src_mean.T)
        }
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
			throw;
		}
        if (losses[0] > losses[1])
        {
            rot = rots[1];
        }            
        else
        {
            rot = rots[0];
        }      
		if (SHOWOUT)
		{
			std::cout << "trans" << trans << std::endl;
            std::cout << "rot" << rot << std::endl;
		}


        std::tuple<torch::Tensor, torch::Tensor> result = std::make_tuple(rot, trans);

        //, trans, scale
        return result;

    }



    using ms = std::chrono::milliseconds;
    using clk = std::chrono::system_clock;
    void LinearBlendSkinning::hybrik(
        const torch::Tensor& pose_skeleton,
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
        const torch::Tensor& lbs_weights,
        const torch::Tensor& restJoints_24)
    {
        if (SHOWOUT)
        {
            std::cout << "LinearBlendSkinning::hybrik" << std::endl;
        }
        batch_size = pose_skeleton.size(0);
        torch::Device device = pose_skeleton.device();
        std::cout << "begin ...................." << std::endl;
        auto begin0 = clk::now();
		
        // 1. Add shape contribution
        //torch::Tensor result = blend_shapes(betas, shapedirs);
        torch::Tensor v_shaped = v_template + blend_shapes(betas, shapedirs);
        if (SHOWOUT)
        {
            std::cout << "[-1,-1,:] :" << v_shaped.sizes() << v_shaped.index({ -1, -1, Slice() }) << std::endl;
        }
        
        torch::Tensor  rest_J = torch::zeros({ batch_size, 29, 3 });
        if (SHOWOUT)
        {
            std::cout << "test_J " << rest_J.sizes() << std::endl;
        }

        //rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
        torch::Tensor test_J_part = rest_J.index({ Slice(), Slice(None,24) });
        if (SHOWOUT)
        {
            std::cout << "test_J_part " << test_J_part.sizes() << std::endl;
        }

        //rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
//         test_J_part = vertices2joints(J_regressor, v_shaped);
//         std::cout << "test_J_part " << test_J_part.sizes() << std::endl;

        rest_J.index({ Slice(), Slice(None,24) }) = vertices2joints(J_regressor, v_shaped);
        rest_J = rest_J.clone().to(m__device);
        if (SHOWOUT)
        {
            std::cout << "test_J[0,22,:] " << rest_J.sizes() << rest_J.index({ 0, 22,Slice() }) << std::endl;
        }
        //print('rest_J[0,[9,19],:]', rest_J[0,[9,19],:])

        //leaf_number = [411, 2445, 5905, 3216, 6617]
// 		std::vector<int> v = { 411, 2445, 5905, 3216, 6617 };
//         torch::Tensor leaf_number = torch::from_blob(v.data(), { 1, 5 });
//         std::cout << "leaf_number " << leaf_number << std::endl;

        torch::Tensor leaf_number = torch::tensor({ 411, 2445, 5905, 3216, 6617 });
        if (SHOWOUT)
        {
            std::cout << "==> leaf_number is:\n" << leaf_number << std::endl;
        }


        torch::Tensor  leaf_vertices = v_shaped.index({ Slice(),leaf_number.data() }).clone();// [:, leaf_number] .clone()
        if (SHOWOUT)
        {
            std::cout << "leaf_vertices " << leaf_vertices.sizes() << std::endl << leaf_vertices << std::endl;
        }


        //rest_J[:, 24:] = leaf_vertices
        rest_J.index({ Slice(), Slice(24,None) }) = leaf_vertices;
        if (SHOWOUT)
        {
            std::cout << "test_J " << rest_J.sizes() << rest_J << std::endl;
            //std::cout << "test_J[0,[9,19],:] " << rest_J.sizes() << rest_J.index({ 0, Slice(9,19),Slice() }) << std::endl;
            std::cout << "test_J[0,26,:] " << rest_J.sizes() << rest_J.index({ 0, 26,Slice() }) << std::endl;
        }

        //# 3. Get the rotation matrics
        torch::Tensor rot_mats = batch_inverse_kinematics_transform(
            pose_skeleton, //global_orient, #phis,
            rest_J.clone(), children, parents//, dtype = dtype, train = train,
            //leaf_thetas = leaf_thetas
        );

        /*
                quat = output.rot_mats.reshape(-1, 4)
        aa = quaternion_to_angle_axis(quat)
        aa = aa.reshape(72)
        */
        if (SHOWOUT)
        {
            std::cout << "rot_mats" << rot_mats.sizes() << std::endl;
        }
        rot_mats = rot_mats.squeeze();// (batch_size * 24, 3, 3));
        if (SHOWOUT)
        {
            std::cout << "rot_mats" << rot_mats.sizes() << rot_mats << std::endl;
        }
        //rot_mats = rot_mats.reshape((batch_size * 24, 3, 3));
        rot_mats = rotmat_to_quat(rot_mats);
        if (SHOWOUT)
        {
            std::cout << "rot_mats" << rot_mats.sizes() << std::endl;
        }
        rot_mats = rot_mats.reshape((1, 24 * 4)).unsqueeze(0);
        if (SHOWOUT)
        {
            std::cout << "rot_mats" << rot_mats.sizes() << std::endl;
        }
        torch::Tensor quat = torch::reshape(rot_mats, { -1, 4 });
        if (SHOWOUT)
        {
            std::cout << "quat" << quat << std::endl;
        }
        quat = quaternion_to_angle_axis(quat);
        
        if (SHOWOUT)
        {
            std::cout << "quat2" << quat << std::endl;
        }
        quat = quat.reshape(72);
        if (SHOWOUT)
        {
            std::cout << "quat3" << quat << std::endl;

        }
        auto end0 = clk::now();
		auto duration = std::chrono::duration_cast<ms>(end0 - begin0);
        std::cout << "Time duration to compute pose: " << (double)duration.count()  << " ms" << std::endl;

        std::vector<person*>  g_persons;

        //joints = (kpts_v[0, [16, 17, 1, 2, 12, 0]]).cpu() //#12: neck, 0 : pelvis

        torch::Tensor joint0 = restJoints_24.index({ 0 });
		if (SHOWOUT)
		{
			std::cout << "joint0 ;" << joint0 << std::endl;
		}
        torch::Tensor b = torch::tensor({ 16, 17, 1, 2, 12, 0 });
        torch::Tensor joints = restJoints_24.index({ 0,{b} }).cpu();// [16] [17] .cpu();
        if (SHOWOUT)
        {
            std::cout << "joints;" << joints << std::endl;
        }

	    torch::Tensor joints3d = pose_skeleton.index({ 0,{b} }).cpu();//[:, [16, 17, 1, 2, 12, 0]])  #   [[5, 2, 12, 9]]
		if (SHOWOUT)
		{
			std::cout << "joints3d;" << joints3d << std::endl;
		}
        std::tuple<torch::Tensor, torch::Tensor> rot_trans = umeyama(joints, joints3d);

        torch::Tensor rot_global = std::get<0>(rot_trans);
        torch::Tensor trans_global = std::get<1>(rot_trans);

        torch::Tensor rot_last = cv2.Rodrigues(rot_global)[0].reshape(1, 3)



// 		joints3d = torch.squeeze(joints3d)
// 			joints3d = np.array(joints3d)

		
        int id = 0;
        torch::Tensor Rh = torch::tensor({ 1.0f, 1.0f, 1.0f });
        torch::Tensor Th = torch::tensor({ 0.3, 0.3, 0.3 });
        torch::Tensor shapes = torch::zeros({10});
        quat = quat.to(torch::kCPU);
        person* p = new person(id, Rh, Th,quat,shapes);
        g_persons.push_back(p);

		 id = 1;
		/*torch::Tensor*/ Rh = torch::tensor({ 0.5f, 0.5f, 0.7f });
		/*torch::Tensor*/ Th = torch::tensor({ 0.8, 0.9, 0.2 });
		/*torch::Tensor*/ shapes = torch::zeros({ 10 });
		quat = quat.to(torch::kCPU);
		person* p2 = new person(id, Rh, Th, quat, shapes);
		g_persons.push_back(p2);




        

        ofstream myfile2("data/000000.json");
        write_persons(g_persons, myfile2);
        myfile2.close();

        //save to json file
		  // print tensor 打印前20 个
		//torch::Tensor grad_bottom_tensor
//         quat = quat.to(torch::kCPU);
//         ofstream  myfile("data/000000.json");
//         //out_text.append('[\n')
//         myfile << "[\n";
// 
//         double* ptr = (double*)quat.data_ptr();
// 		for (size_t i = 0; i < 72; i++) {
//             try
//             {
//                 //std::cout << *((ptr + i)) << std::endl;
//                 myfile << *((ptr + i));
//                 myfile << ", ";
// 
//             }
//             catch (const std::exception& e)
//             {
//                 std::cout << e.what() << std::endl;
//                 throw;
//             }
// 			
// 		}
//         myfile << "]\n";
//         myfile.close();

    }




    torch::Tensor LinearBlendSkinning::rotmat_to_quat(torch::Tensor& rotation_matrix)
    {
        //assert rotation_matrix.shape[1:] == (3, 3)
        if (SHOWOUT)
        {
            std::cout << "rotation_matrix" << rotation_matrix.sizes() << std::endl;
        }
        torch::Tensor rot_mat;
        try
        {
            rot_mat = torch::reshape(rotation_matrix, { -1, 3, 3 });
        }
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			throw;
		}
        if (SHOWOUT)
        {
            std::cout << "rot_mat" << rot_mat.sizes() << std::endl;
        }

        torch::Tensor hom = torch::tensor({ 0.0, 0.0, 1.0 });// , dtype = torch.float32,
        hom = hom.to(torch::kFloat64);
        if (SHOWOUT)
        {
            std::cout << "hom" << hom << std::endl;
        }
        hom = hom.reshape({ 1, 3, 1 }).expand({ rot_mat.size(0), -1, -1 });
        if (SHOWOUT)
        {
            std::cout << "hom" << hom << std::endl;
        }
        rot_mat = rot_mat.to(m__device);
        hom = hom.to(m__device);
        rotation_matrix = rotation_matrix.to(m__device);
        try
        {
            rotation_matrix = torch::cat({ rot_mat, hom }, 2);
        }
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			throw;
		}
        if (SHOWOUT)
        {
            std::cout << "rotation_matrix" << rotation_matrix << std::endl;
        }
// 
//         hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
//         rotation_matrix = torch.cat([rot_mat, hom], dim = -1)
// 
//         quaternion = rotation_matrix_to_quaternion(rotation_matrix)

        torch::Tensor quaternion = rotation_matrix_to_quaternion(rotation_matrix);
        return quaternion;//暂时 quaternion;

    }
    
    void LinearBlendSkinning::write_persons(std::vector<person*> persons, ofstream& file)
    {
        file << "[\n";
        int num = persons.size();
        int index = 0;
        for (std::vector<person*>::iterator iter = persons.begin(); iter != persons.end(); iter++)
        {
            person *p = *iter;
            write_json(file, p->m_id, p->m_Rh, p->m_Th, p->m_poses, p->m_shapes);
            if (index != num -1)
            {
                file << ",";
                index++;
            }
        }
        file << "]";

    }

    person::person(int id, torch::Tensor Rh, torch::Tensor Th, torch::Tensor poses, torch::Tensor shapes) :
        m_id(id), m_Rh(Rh), m_Th(Th), m_poses(poses), m_shapes(shapes)
    {

    }

    

    //=============================================================================
} // namespace smpl
//=============================================================================
