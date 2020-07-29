#ifndef CATEGORICAL_H
#define CATEGORICAL_H

class Categorical
{
public:
    Categorical(torch::Tensor& probs);
    ~Categorical();

    std::vector<int64_t> Extended_Shape(torch::ArrayRef<int64_t> sample_shape);
    torch::Tensor Sample(torch::ArrayRef<int64_t> sample_shape = {});
    torch::Tensor Entropy();
    torch::Tensor Log_Prob(torch::Tensor& actprob);
    torch::Tensor Get_Logits();
    torch::Tensor Get_Probs();

private:
    int 					m_num_events;
    std::vector<int64_t> 	m_batch_shape, m_event_shape;
    torch::Tensor 			m_probs, m_logits, m_param;
};

#endif
