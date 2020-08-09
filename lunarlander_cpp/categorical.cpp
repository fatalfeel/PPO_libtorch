#include "torch/torch.h"
using namespace torch;
#include "categorical.h"

Categorical::Categorical(torch::Tensor probs)
{
	torch::Tensor param;

	m_batch_shape 	= {};
	m_probs 		= probs / probs.sum(-1, true);
	m_probs 		= this->m_probs.clamp(1.21e-7, 1.0 - 1.21e-7);
	m_logits 		= torch::log(this->m_probs);

	param 			= probs;
    m_num_events 	= param.size(-1);

    if (param.dim() > 1)
    {
        m_batch_shape = param.sizes().vec();
        m_batch_shape.resize(m_batch_shape.size() - 1);
    }
}

Categorical::~Categorical()
{
}

//refer to /usr/lib/python3.7/site-packages/torch/distributions/distribution.py
std::vector<int64_t> Categorical::Extended_Shape(c10::ArrayRef<int64_t> shape)
{
    std::vector<int64_t> output_shape;

    output_shape.insert(output_shape.end(), shape.begin(),  shape.end());
    output_shape.insert(output_shape.end(), m_batch_shape.begin(), m_batch_shape.end());
    //output_shape.insert(output_shape.end(),	m_event_shape.begin(), m_event_shape.end());

    return output_shape;
}

//refer to /usr/lib/python3.7/site-packages/torch/distributions/categorical.py
torch::Tensor Categorical::Sample(torch::ArrayRef<int64_t> shape)
{
	std::vector<int64_t>	vecshape;
	std::vector<int64_t> 	param_shape;
	Tensor 					exp_probs;
	Tensor 					probs_2d;
	Tensor 					sample_2d;

	vecshape	= Extended_Shape(shape);
	param_shape = vecshape;
    param_shape.insert(param_shape.end(), {m_num_events});

    exp_probs = m_probs.expand(param_shape);

    if (m_probs.dim() == 1 || m_probs.size(0) == 1)
        probs_2d = exp_probs.view({-1, m_num_events});
    else
        probs_2d = exp_probs.contiguous().view({-1, m_num_events});

    sample_2d = torch::multinomial(probs_2d, 1, true);

    return sample_2d.contiguous().view(vecshape); //out is int64_t
}

torch::Tensor Categorical::Entropy()
{
	Tensor p_log_p = m_logits * m_probs;
    return -p_log_p.sum(-1).to(torch::kFloat64);
}

torch::Tensor Categorical::Log_Prob(torch::Tensor actprob)
{
	std::vector<Tensor>	broadcasted_tensors;

	actprob 			= actprob.to(torch::kInt64).unsqueeze(-1);
    broadcasted_tensors = torch::broadcast_tensors({actprob, m_logits});
    actprob 			= broadcasted_tensors[0];
    actprob 			= actprob.narrow(-1, 0, 1);

    return broadcasted_tensors[1].gather(-1, actprob).squeeze(-1).to(torch::kFloat64);
}

