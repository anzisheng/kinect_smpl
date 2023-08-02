#pragma once
#ifndef __SMPLCAM__
#define __SMPLCAM__

#include <torch/torch.h>
using namespace torch;
class smplcam
{
public:
	smplcam();
	//void call_forward();
private:
	int  m_smpl_dtype;
	torch::Tensor m_h36m_jregressor;
};


#endif