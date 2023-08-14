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
#include "definition/def.h"
#include "toolbox/Exception.h"
#include "toolbox/TorchEx.hpp"
#include "smpl/LinearBlendSkinning.h"
#include "torch/script.h"
using namespace torch::indexing;
#include <exception>
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

        std::cout << "betas shape:" << betas<< std::endl;
        std::cout << "shape_disps[-1,:,-1]:" << shape_disps.index({-1, Slice(),-1}) << std::endl;
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
        //print("blend_shape[��,-1,:]", blend_shape[:, -1, :])
        std::cout << "blend_shape shape[��,-1,:]:" << blend_shape.index({ Slice(), -1, Slice()}) << std::endl;
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
        std::cout << "rel_rest_pose :" << rel_rest_pose.sizes() << std::endl;
        //rel_rest_pose[:, 1 : ] -= rest_pose[:, parents[1:]].clone();

 		std::vector<int> parents_exclude_0_v = { 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11 };
        torch::Tensor parents_exclude_0 = torch::tensor(parents_exclude_0_v);// .data(), { 29 }).toType(torch::kInt8);
        parents_exclude_0 = parents_exclude_0.to(m__device);
        std::cout << "parents_exclude_0:" << parents_exclude_0.sizes() << parents_exclude_0.device()<< parents_exclude_0 << std::endl;

        //torch::Tensor par = parents.index({ Slice(1,None) });                
        //std::cout << "par:" << par.sizes() << par << std::endl;

        //torch::Tensor b = torch::tensor({ 0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2 });
//         torch::Tensor temp = rest_pose.index({ Slice(), {parents_exclude_0}});
//         std::cout << "temp :" << temp.sizes() << std::endl;

        //rel_rest_pose.index({ Slice(),Slice(1),Slice() }) -= rest_pose.index({ Slice(), parents.index({Slice(1,None)}) });// .clone();
        // 
        rel_rest_pose.index({ Slice(),Slice(1),Slice() }) -= rest_pose.index({ Slice(), {parents_exclude_0} });
        std::cout << "rel_rest_pose:" << rel_rest_pose.sizes()<< rel_rest_pose << std::endl;
        std::cout << "rel_rest_pose.index({ Slice(),Slice(1),Slice() }):" << rel_rest_pose.index({ Slice(),Slice(1),Slice() }).sizes() << rel_rest_pose.index({ Slice(),Slice(1),Slice() }) << std::endl;
        

        rel_rest_pose = torch::unsqueeze(rel_rest_pose, -1);
        std::cout << "rel_rest_pose:" << rel_rest_pose.sizes() << std::endl; //torch.Size([1, 29, 3, 1])


		//# rotate the T pose
        torch::Tensor rotate_rest_pose = torch::zeros_like(rel_rest_pose);
        std::cout << "rotate_rest_pose:" << rotate_rest_pose.sizes() << std::endl;

		//# set up the root
        //rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]
        rotate_rest_pose.index({ Slice(), 0 }) = rel_rest_pose.index({ Slice(),0 });
        std::cout << "rotate_rest_pose:" << rotate_rest_pose.sizes() << std::endl;


// 		rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim = -1).detach()
//		rel_pose_skeleton[:, 1 : ] = rel_pose_skeleton[:, 1 : ] - rel_pose_skeleton[:, parents[1:]].clone()
// 		rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        //torch::Tensor b2 = torch::tensor({ 0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2 });

        torch::Tensor rel_pose_skeleton = torch::unsqueeze(pose_skeleton.clone(), -1).detach();
		torch::Tensor temp = torch::squeeze(rel_pose_skeleton);
		std::cout << "temp:" << temp.sizes() << temp << std::endl;

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

        std::cout << "rel_pose_skeleton:" << rel_pose_skeleton.sizes() << rel_pose_skeleton << std::endl;

        /*torch::Tensor*/ temp = torch::squeeze(rel_pose_skeleton);
        std::cout << "temp:" << temp.sizes() << temp << std::endl;


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

        std::cout << "final_pose_skeleton: " << final_pose_skeleton << std::endl;

		temp = torch::squeeze(final_pose_skeleton);
        std::cout << "temp[[3]]:" << temp.sizes() << temp.index({ 3,Slice()}) << std::endl;
        std::cout << "temp[[13]]:" << temp.sizes() << temp.index({ 13,Slice() }) << std::endl;
        std::cout << "temp[[26]]:" << temp.sizes() << temp.index({ 26,Slice() }) << std::endl;


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
        std::cout << "parents :" << parents << std::endl;
        std::cout << "children :" << children << std::endl;

        torch::Tensor global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children);
        //global_orient_mat = global_orient_mat.to(torch::kFloat64);
        std::cout << "torglobal_orient_mat:" << global_orient_mat.sizes() << global_orient_mat << std::endl;

        
        
        
        
        //��ע�Ĵ���

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
            std::cout << parents.size(0) << children.index({ i }).item();
            if(children.index({ i }).item() == -1)
            {

            }
            else if (child_indices[i].size() == 3)//: #children[i] == -3)
            {
                //ֻ�е�i =9,ʱchild_indices[9]����3���˴�Ӳ���룬��ʱparents[i] Ϊ6
				// three children
// 				rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
// 					rot_mat_chain[parents[i]],
// 					rel_rest_pose[:, i]
// 				)
                std::cout << "rotate_rest_pose[:,0]:" << rotate_rest_pose.sizes() << rotate_rest_pose.index({ Slice(),0,Slice(),Slice() }) << std::endl;
                rotate_rest_pose = rotate_rest_pose.to(torch::kFloat64);

                std::cout << "rot_mat_chain[6]:" << rot_mat_chain[6].type()  << std::endl;
                rel_rest_pose = rel_rest_pose.to(torch::kFloat64);
                std::cout << "rel_rest_pose.index({ Slice(),9 }):" << rel_rest_pose.index({ Slice(),9 }).type() << std::endl;

                //�˲��ȵ�i = 9,�ٲ��ԣ���Ϊ��ʱrot_mat_chain��Ԫ�ظ����Ź�[6]��
                try 
                {
                    rotate_rest_pose.index({ Slice(),i }) = rotate_rest_pose.index({ Slice(), 6 }) + torch::matmul(rot_mat_chain[6], rel_rest_pose.index({ Slice(),9 }));// [:, i] );// [, parents[i]]
                }
				catch (std::exception& e)
				{
					std::cout << e.what() << std::endl;
					throw;
				}

                std::cout << "rotate_rest_pose.index({ Slice(),i }):" << rotate_rest_pose.index({ Slice(),i }).sizes() << rotate_rest_pose.index({ Slice(),i }) << std::endl;
                

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

// 				rot_mat = batch_get_3children_orient_svd(
// 					children_final_loc, children_rest_loc,
// 					rot_mat_chain[parents[i]], spine_child, dtype)





            }
            else if (child_indices[i].size() == 1)
            {
                //�󲿷ֶ�ִ�д�������
				//only one child
// 				child = child_indices[i][0]
                int child = child_indices[i][0];

                torch::Tensor child_rest_loc = rel_rest_pose.index({ Slice(),child });// [:, child]
                child_rest_loc = child_rest_loc.to(torch::kFloat64);
                std::cout << "child_rest_loc " << child_rest_loc << std::endl;
                torch::Tensor child_final_loc = rel_pose_skeleton.index({ Slice(), child});// [:, child]
                std::cout << "child_final_loc " << child_final_loc << std::endl;
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
                    std::cout << "child_final_loc " << child_final_loc << std::endl;
                }
				catch (std::exception& e)
				{
					std::cout << e.what() << std::endl;
					throw;
				}

                //child_final_norm = torch.norm(child_final_loc, dim = 1, keepdim = True)
                torch::Tensor child_final_norm = torch::norm(child_final_loc, 2, 1, true);
                std::cout << "child_final_norm " << child_final_norm << std::endl;
                //child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
                torch::Tensor  child_rest_norm = torch::norm(child_rest_loc, 2, 1, true);
                std::cout << "child_rest_norm " << child_rest_norm << std::endl;


                //axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
                torch::Tensor axis = torch::cross(child_rest_loc, child_final_loc, 1);
                std::cout << "axis" << axis << std::endl;

                //axis_norm = torch.norm(axis, dim=1, keepdim=True)
                torch::Tensor axis_norm = torch::norm(axis,2, 1,true);
                std::cout << "axis_norm" << axis_norm << std::endl;

                //temp = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True)
                torch::Tensor temp = torch::sum(child_rest_loc * child_final_loc, 1, true);
                std::cout << "temp" << temp << std::endl;


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
                std::cout << src << std::endl;
                //rot_mat = cv2.Rodrigues(aa.detach().cpu().numpy())[0]
                
				//Mat auxRinv = Mat::eye(3, 3, CV_32F);
				//Rodrigues(Rc1, auxRinv);
                //rot_mat = cv.Rodrigues(aa.detach().cpu().numpy())[0]
                cv::Mat dst;
                cv::Rodrigues(src, dst);
                std::cout << dst << std::endl;
                torch::Tensor rot_mat = torch::from_blob(dst.data, { 3, 3 }, torch::kFloat64);
                std::cout << "rot_mat: " << rot_mat.sizes() << rot_mat << std::endl;

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
                if (SHOWOUT)
                {
                    torch::Tensor tempp = rot_mat_chain[in];

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
        }

		
        





        return rel_rest_pose;//temperary


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
        std::cout << "rel_rest_pose temp all:" << temp << std::endl;

        torch::Tensor rest_mat = torch::zeros({9});
        torch::Tensor target_mat = torch::zeros({ 9 });
        int i = 0;
        for (std::vector<int>::iterator itr =pelvis_child.begin(); itr != pelvis_child.end(); itr++ )
        {
            
            int child = *itr;
            //rest_mat.index({ Slice(i * 3, i * 3 + 3) }).print();
            torch::Tensor temp = rel_rest_pose.index({ Slice(),child }).clone();//.print();
            temp = torch::squeeze(temp);
            std::cout << "temp:" << temp << std::endl;

			torch::Tensor temp2 = rel_pose_skeleton.index({ Slice(),child }).clone();
			temp2 = torch::squeeze(temp2);
			std::cout << "temp2:" << temp2 << std::endl;

            //std::cout << "rel_rest_pose:" << rel_rest_pose << std::endl;
            rest_mat.index({ Slice(i * 3, i * 3 + 3) }) = temp;// rel_rest_pose.index({ Slice(),child }).clone();// [:, child] .clone()
            target_mat.index({ Slice(i * 3, i * 3 + 3) }) = temp2;// rel_rest_pose.index({ Slice(),child }).clone();// [:, child] .clone()

            //rest_mat.print();
            std::cout << "rest_mat:" << rest_mat << std::endl;
            std::cout << "target_mat:" << target_mat<< std::endl;

            //rel_rest_pose.index({ Slice(),child }).print();
            i++;
        }
        
        rest_mat = torch::reshape(rest_mat, { 1,3,3 });// .transpose(0, 1);
        rest_mat = torch::squeeze(rest_mat).transpose(0,1);
        rest_mat = torch::unsqueeze(rest_mat, 0);
        std::cout << "rest_mat:" << rest_mat.sizes()<< rest_mat << std::endl;
        target_mat = torch::reshape(target_mat, { 1,3,3 });
        target_mat = torch::squeeze(target_mat).transpose(0, 1);
        target_mat = torch::unsqueeze(target_mat, 0);
		std::cout << "target_mat:" << target_mat.sizes() << target_mat << std::endl;


        //S = rest_mat.bmm(target_mat.transpose(1, 2))
        torch::Tensor S = rest_mat.bmm(target_mat.transpose(1, 2));
        std::cout << "S:" << S.sizes() << S << std::endl;

        
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
        std::cout << "V:" << V.sizes() << V.device()<<V<< std::endl;

        U = U.to(m__device);
        V = V.to(m__device);
        std::cout << "V:" << V.sizes() << V.device() << V << std::endl;
//		U = U.to(device = device)
//		V = V.to(device = device)



        torch::Tensor rot_mat = torch::zeros_like(S);
        torch::Tensor det_u_v = torch::det(torch::bmm(V, U.transpose(1, 2)));
        std::cout << "det_u_v:" << det_u_v.sizes() << det_u_v.device() << det_u_v << std::endl;

        torch::Tensor det_modify_mat = torch::eye(3, U.device()).unsqueeze(0);// .expand(U.size(0), -1, -1).clone()
        std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;
        det_modify_mat.index_put_({ Slice(),2,2 }, det_u_v);// = det_u_v;
        std::cout << "det_modify_mat:" << det_modify_mat.sizes() << det_modify_mat.device() << det_modify_mat << std::endl;

        //rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))
        torch::Tensor rot_mat_non_zero = torch::bmm(torch::bmm(V, det_modify_mat), U.transpose(1, 2));
        std::cout << "rot_mat_non_zero:" << rot_mat_non_zero.sizes() << rot_mat_non_zero.device() << rot_mat_non_zero << std::endl;
        return rot_mat_non_zero;
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

            target = torch::matmul(
                rot_mat_chain_parent.transpose(1, 2), target);
            //std::cout << rot_mat_chain_parent.
            target_mat.push_back(target);
            rest_mat.push_back(templates);
        }

// 		rest_mat = torch.cat(rest_mat, dim = 2)
        rest_mat = torch::cat(rest_mat,2);
        //rest_mat = torch::stack(rest_mat, 1)
        
        target_mat = torch::cat(target_mat, 2);
         
        torch::Tensor S = rest_mat.bmm(target_mat.transpose(1, 2));

        /*
		*     device = S.device
	        U, _, V = torch.svd(S.cpu())
	        U = U.to(device=device)
	        V = V.to(device=device)
        */

        /*
        * 
		*     # rot_mat = torch.bmm(V, U.transpose(1, 2))
	det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
	det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
	det_modify_mat[:, 2, 2] = det_u_v
	rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

        */




        torch::Tensor rot_mat;

    }


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
        const torch::Tensor& lbs_weights)
    {
        std::cout << "LinearBlendSkinning::hybrik" << std::endl;
        batch_size = pose_skeleton.size(0);
        torch::Device device = pose_skeleton.device();

        // 1. Add shape contribution
        //torch::Tensor result = blend_shapes(betas, shapedirs);
        torch::Tensor v_shaped = v_template + blend_shapes(betas, shapedirs);
        //std::cout << "v_template :" << v_template.sizes() << v_template <<std::endl;
        std::cout << "[-1,-1,:] :" << v_shaped.sizes() << v_shaped.index({-1, -1, Slice()}) << std::endl;
        //print("v_shaped[-1,-1,:]", v_shaped[-1,-1,:])

        //rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
        torch::Tensor  rest_J = torch::zeros({ batch_size, 29, 3 });
        std::cout << "test_J " << rest_J.sizes() << std::endl;

        //rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
        torch::Tensor test_J_part = rest_J.index({ Slice(), Slice(None,24) });

        std::cout << "test_J_part " << test_J_part.sizes() << std::endl;

        //rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
//         test_J_part = vertices2joints(J_regressor, v_shaped);
//         std::cout << "test_J_part " << test_J_part.sizes() << std::endl;

        rest_J.index({ Slice(), Slice(None,24) }) = vertices2joints(J_regressor, v_shaped);
        rest_J = rest_J.clone().to(m__device);
        std::cout << "test_J[0,22,:] " << rest_J.sizes() << rest_J.index({0, 22,Slice()}) << std::endl;
        //print('rest_J[0,[9,19],:]', rest_J[0,[9,19],:])

        //leaf_number = [411, 2445, 5905, 3216, 6617]
// 		std::vector<int> v = { 411, 2445, 5905, 3216, 6617 };
//         torch::Tensor leaf_number = torch::from_blob(v.data(), { 1, 5 });
//         std::cout << "leaf_number " << leaf_number << std::endl;

        torch::Tensor leaf_number = torch::tensor({ 411, 2445, 5905, 3216, 6617 });
		std::cout << "==> leaf_number is:\n" << leaf_number << std::endl;


        torch::Tensor  leaf_vertices = v_shaped.index({ Slice(),leaf_number.data() }).clone();// [:, leaf_number] .clone()
        std::cout << "leaf_vertices " << leaf_vertices.sizes()<< std::endl << leaf_vertices <<std::endl;


        //rest_J[:, 24:] = leaf_vertices
        rest_J.index({ Slice(), Slice(24,None) }) = leaf_vertices;
        std::cout << "test_J " << rest_J.sizes() << rest_J<< std::endl;
        //std::cout << "test_J[0,[9,19],:] " << rest_J.sizes() << rest_J.index({ 0, Slice(9,19),Slice() }) << std::endl;
        std::cout << "test_J[0,26,:] " << rest_J.sizes() << rest_J.index({ 0, 26,Slice() }) << std::endl;

        //# 3. Get the rotation matrics
        torch::Tensor rot_mats = batch_inverse_kinematics_transform(
            pose_skeleton, //global_orient, #phis,
            rest_J.clone(), children, parents//, dtype = dtype, train = train,
            //leaf_thetas = leaf_thetas
        );

    }



    //=============================================================================
} // namespace smpl
//=============================================================================
