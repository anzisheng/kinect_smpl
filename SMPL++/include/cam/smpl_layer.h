#pragma once
#ifndef __SMPL_LAYER__
#define __SMPL_LAYER__
#include <string>
using namespace std;
class smpl_layer
{
public:
	smpl_layer();
private:
	static const int NUM_JOINTS = 23;
	static const int NUM_BODY_JOINTS = 23;
	static const int NUM_BETAS = 10;
	std::string JOINT_NAMES[29];
	std::string LEAF_NAMES[5];
	static const int root_idx_17 = 0;
	static const int root_idx_smpl = 0;

// 	std::string[29] JOINT_NAMES = { 
// 		'pelvis', 'left_hip', 'right_hip',      // 2
// 		'spine1', 'left_knee', 'right_knee',    // 5
// 		'spine2', 'left_ankle', 'right_ankle',  // 8
// 		'spine3', 'left_foot', 'right_foot',    // 11
// 		'neck', 'left_collar', 'right_collar',  // 14
// 		'jaw',                                  // 15
// 		'left_shoulder', 'right_shoulder'
// 		,      // 17
// 		'left_elbow', 'right_elbow',            // 19
// 		'left_wrist', 'right_wrist',            // 21
// 		'left_thumb', 'right_thumb',            // 23
// 		'head', 'left_middle', 'right_middle',  // 26
// 		'left_bigtoe', 'right_bigtoe'           // 28
// 
// 	};
	

};

#endif