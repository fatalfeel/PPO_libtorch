#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <limits>
#include <torch/torch.h>
using namespace torch;
#include "normaldistribute.h"

NormalDistribute::NormalDistribute(torch::Tensor mu, torch::Tensor scale)
{
	std::vector<Tensor> broadcasted_tensors;

	broadcasted_tensors = torch::broadcast_tensors({mu, scale});
    m_mu 				= broadcasted_tensors[0];
    m_scale	 			= broadcasted_tensors[1];
    m_batch_shape 		= m_mu.sizes().vec();
    //m_event_shape 	= {};
}

//refer to /usr/lib/python3.7/site-packages/torch/distributions/distribution.py
std::vector<int64_t> NormalDistribute::Extended_Shape(ArrayRef<int64_t> shape)
{
    std::vector<int64_t> output_shape;

    output_shape.insert(output_shape.end(), shape.begin(),  shape.end());
    output_shape.insert(output_shape.end(), m_batch_shape.begin(), m_batch_shape.end());
    //output_shape.insert(output_shape.end(),	m_event_shape.begin(), m_event_shape.end());

    return output_shape;
}

//refer to /usr/lib/python3.7/site-packages/torch/distributions/normal.py
torch::Tensor NormalDistribute::Sample(c10::ArrayRef<int64_t> shape)
{
	std::vector<int64_t>	vecshape;
	Tensor					sample;
	//NoGradGuard 			nograd; //declare here torch backward() will fail

	vecshape 			= Extended_Shape(shape);
	NoGradGuard nograd 	= NoGradGuard();
	sample				= at::normal(m_mu.expand(vecshape), m_scale.expand(vecshape));

	return sample.to(torch::kFloat64);
}

torch::Tensor NormalDistribute::Entropy()
{
	Tensor entropy = (0.5 + 0.5 * std::log(2 * M_PI) + torch::log(m_scale));
	return entropy.to(torch::kFloat64);
}

//refer to /usr/lib/python3.7/site-packages/torch/distributions/normal.py
torch::Tensor NormalDistribute::Log_Prob(torch::Tensor value)
{
	Tensor 	variance 	= m_scale.pow(2);
	Tensor	log_scale	= m_scale.log();
	Tensor	logprob		= -((value - m_mu).pow(2)) / (2 * variance) - log_scale - std::log(std::sqrt(2 * M_PI));

	return logprob.to(torch::kFloat64);
}

