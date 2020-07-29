#include "torch/torch.h"
using namespace torch;
#include "categorical.h"

Categorical::Categorical(torch::Tensor probs)
{
	//m_batch_shape = {};
	//m_event_shape = {};

	this->m_probs = probs / probs.sum(-1, true);
	// 1.21e-7 is used as the epsilon to match PyTorch's Python results as closely
	// as possible
	this->m_probs 	= this->m_probs.clamp(1.21e-7, 1.0 - 1.21e-7);
	this->m_logits 	= torch::log(this->m_probs);


    /*m_param 		= probs;
    m_num_events 	= m_param.size(-1);
    if (m_param.dim() > 1)
    {
        m_batch_shape = m_param.sizes().vec();
        m_batch_shape.resize(m_batch_shape.size() - 1);
    }*/
	m_num_events 	= probs.size(-1);
}

Categorical::~Categorical()
{
}

/*std::vector<int64_t> Categorical::Extended_Shape(c10::ArrayRef<int64_t> sample_shape)
{
    std::vector<int64_t> output_shape;
    output_shape.insert(output_shape.end(), sample_shape.begin(),  sample_shape.end());
    output_shape.insert(output_shape.end(), m_batch_shape.begin(), m_batch_shape.end());
    output_shape.insert(output_shape.end(),	m_event_shape.begin(), m_event_shape.end());
    return output_shape;
}*/

torch::Tensor Categorical::Sample(torch::ArrayRef<int64_t> shape)
{
	/*std::vector<int64_t> ext_sample_shape	= Extended_Shape(sample_shape);
	std::vector<int64_t> param_shape 		= ext_sample_shape;*/
	std::vector<int64_t> 	ext_sample_shape;
	std::vector<int64_t> 	param_shape;
	Tensor 					exp_probs;
	Tensor 					probs_2d;
	ensor 					sample_2d;

	ext_sample_shape.insert(ext_sample_shape.end(), shape.begin(), shape.end());

	param_shape = ext_sample_shape;
    param_shape.insert(param_shape.end(), {m_num_events});

    exp_probs = m_probs.expand(param_shape);

    if (m_probs.dim() == 1 || m_probs.size(0) == 1)
        probs_2d = exp_probs.view({-1, m_num_events});
    else
        probs_2d = exp_probs.contiguous().view({-1, m_num_events});

    sample_2d = torch::multinomial(probs_2d, 1, true);

    return sample_2d.contiguous().view(ext_sample_shape);
}

torch::Tensor Categorical::Entropy()
{
	Tensor p_log_p = m_logits * m_probs;
    return -p_log_p.sum(-1);
}

torch::Tensor Categorical::Log_Prob(torch::Tensor& actprob)
{
	torch::Tensor local_prob = actprob;

	local_prob = local_prob.to(torch::kLong).unsqueeze(-1);
    auto broadcasted_tensors = torch::broadcast_tensors({local_prob, m_logits});
    local_prob = broadcasted_tensors[0];
    local_prob = local_prob.narrow(-1, 0, 1);
    return broadcasted_tensors[1].gather(-1, local_prob).squeeze(-1);
}

