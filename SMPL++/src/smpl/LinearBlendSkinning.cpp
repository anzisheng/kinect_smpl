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

        std::cout << "betas shape:" << betas.sizes() << std::endl;
        std::cout << "shape_disps shape:" << shape_disps.sizes() << std::endl;

        torch::Tensor blend_shape;
        std::cout << "blend_shape shape:" << blend_shape.sizes() << std::endl;
        try
        {
            //at::TensorList t = { betas, shape_disps };
            blend_shape = torch::einsum("bl, mkl->bmk", { betas, shape_disps });
            std::cout << "blend_shape shape:" << blend_shape.sizes() << std::endl;
        }
		catch (std::exception& e)
		{
            std::cout << e.what() << std::endl;
			throw;
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
        std::cout << "rel_rest_pose :" << rel_rest_pose.sizes() << std::endl;
        //rel_rest_pose[:, 1 : ] -= rest_pose[:, parents[1:]].clone();


        torch::Tensor par = parents.index({ Slice(1,None) });                
        std::cout << "par:" << par.sizes() << par << std::endl;

        torch::Tensor b = torch::tensor({ 0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2,0, 1, 3, 2 });
        torch::Tensor temp = rest_pose.index({ Slice(), {b}});
        std::cout << "temp :" << temp.sizes() << std::endl;

        //rel_rest_pose.index({ Slice(),Slice(1),Slice() }) -= rest_pose.index({ Slice(), parents.index({Slice(1,None)}) });// .clone();
        // 
        rel_rest_pose.index({ Slice(),Slice(1),Slice() }) = rest_pose.index({ Slice(), {b} });        
        std::cout << "rel_rest_pose:" << rel_rest_pose.sizes() << std::endl;

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

        torch::Tensor rel_pose_skeleton = torch::unsqueeze(pose_skeleton.clone(), -1).detach();
        rel_pose_skeleton.index({ Slice(),1,Slice() }) = rel_pose_skeleton.index({ Slice(),1, Slice() }) - rel_pose_skeleton.index({ Slice(),{b} }).clone();
        rel_pose_skeleton.index({ Slice(),0 }) = rel_rest_pose.index({ Slice(),0 });

// 		# the predicted final pose
// 			final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim = -1)
// 			final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0 : 1] + rel_rest_pose[:, 0 : 1]
        
        torch::Tensor final_pose_skeleton = torch::unsqueeze(pose_skeleton.clone(), -1);
        final_pose_skeleton = final_pose_skeleton - final_pose_skeleton.index({ Slice(), Slice(0,1) }) + rel_rest_pose.index({ Slice(), Slice(0,1) });//


        rel_rest_pose = rel_rest_pose;
        //rel_pose_skeleton = rel_pose_skeleton;
        final_pose_skeleton = final_pose_skeleton;
        rotate_rest_pose = rotate_rest_pose;


        /*
		global_orient_mat = batch_get_pelvis_orient_svd(
		rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
        */








        return rel_rest_pose;//temperary





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
        //device = pose_skeleton.device;

        // 1. Add shape contribution
        //torch::Tensor result = blend_shapes(betas, shapedirs);
        torch::Tensor v_shaped = v_template + blend_shapes(betas, shapedirs);
        std::cout << "v_shaped :" << v_shaped.sizes() << std::endl;

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
        std::cout << "test_J " << rest_J.sizes() << rest_J << std::endl;

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
