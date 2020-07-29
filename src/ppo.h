#ifndef CPPO_H
#define CPPO_H

class CPPO
{
public:
	CPPO();
    ~CPPO();

    void Predict_Reward(torch::Tensor& next_state, GameContent& gamedata);
    void Train_Update(GameContent& gamedata);

private:
    int							m_train_epochs;
    double      				m_lr;
	double 						m_gamma;
    double 						m_eps_clip;
    std::tuple<double,double>	m_betas;

	ActorCritic*  				m_policy_ac;
    optim::Adam* 				m_optimizer;
    //torch::Device 			m_device;
};

#endif
