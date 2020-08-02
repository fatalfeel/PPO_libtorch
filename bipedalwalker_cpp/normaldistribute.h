#ifndef NORMALDISTRIBUTE_H

class NormalDistribute
{
public:
    NormalDistribute(torch::Tensor mu, torch::Tensor scale);
    std::vector<int64_t> 	Extended_Shape(ArrayRef<int64_t> shape);
    torch::Tensor 			Entropy();
    torch::Tensor 			Log_Prob(torch::Tensor value);
    torch::Tensor 			Sample(ArrayRef<int64_t> shape = {});

private:
    std::vector<int64_t> 	m_batch_shape;
    torch::Tensor 			m_mu, m_scale;
};

#endif
